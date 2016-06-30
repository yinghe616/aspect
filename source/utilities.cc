/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include <aspect/global.h>
#include <aspect/utilities.h>

#include <deal.II/base/std_cxx11/array.h>
#include <deal.II/base/point.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/exceptions.h>

#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/chunk.h>

#include <fstream>
#include <string>
#include <sys/stat.h>
#include <errno.h>

namespace aspect
{
  /**
   * A namespace for utility functions that might be used in many different
   * places to prevent code duplication.
   */
  namespace Utilities
  {
    using namespace dealii;

    /**
     * Split the set of DoFs (typically locally owned or relevant) in @p whole_set into blocks
     * given by the @p dofs_per_block structure.
     */
    void split_by_block (const std::vector<types::global_dof_index> &dofs_per_block,
                         const IndexSet &whole_set,
                         std::vector<IndexSet> &partitioned)
    {
      const unsigned int n_blocks = dofs_per_block.size();
      partitioned.clear();

      partitioned.resize(n_blocks);
      types::global_dof_index start = 0;
      for (unsigned int i=0; i<n_blocks; ++i)
        {
          partitioned[i] = whole_set.get_view(start, start + dofs_per_block[i]);
          start += dofs_per_block[i];
        }
    }

    template <int dim>
    std_cxx11::array<double,dim>
    spherical_coordinates(const Point<dim> &position)
    {
      std_cxx11::array<double,dim> scoord;

      scoord[0] = position.norm(); // R
      scoord[1] = std::atan2(position(1),position(0)); // Phi
      if (scoord[1] < 0.0)
        scoord[1] += 2.0*numbers::PI; // correct phi to [0,2*pi]
      if (dim==3)
        {
          if (scoord[0] > std::numeric_limits<double>::min())
            scoord[2] = std::acos(position(2)/scoord[0]);
          else
            scoord[2] = 0.0;
        }
      return scoord;
    }

    template <int dim>
    Point<dim>
    cartesian_coordinates(const std_cxx11::array<double,dim> &scoord)
    {
      Point<dim> ccoord;

      switch (dim)
        {
          case 2:
          {
            ccoord[0] = scoord[0] * std::cos(scoord[1]); // X
            ccoord[1] = scoord[0] * std::sin(scoord[1]); // Y
            break;
          }
          case 3:
          {
            ccoord[0] = scoord[0] * std::sin(scoord[2]) * std::cos(scoord[1]); // X
            ccoord[1] = scoord[0] * std::sin(scoord[2]) * std::sin(scoord[1]); // Y
            ccoord[2] = scoord[0] * std::cos(scoord[2]); // Z
            break;
          }
          default:
            Assert (false, ExcNotImplemented());
            break;
        }

      return ccoord;
    }


    template <int dim>
    std_cxx11::array<Tensor<1,dim>,dim-1>
    orthogonal_vectors (const Tensor<1,dim> &v)
    {
      Assert (v.norm() > 0,
              ExcMessage ("This function can not be called with a zero "
                          "input vector."));

      std_cxx11::array<Tensor<1,dim>,dim-1> return_value;
      switch (dim)
        {
          case 2:
          {
            // create a direction by swapping the two coordinates and
            // flipping one sign; this is orthogonal to 'v' and has
            // the same length already
            return_value[0][0] = v[1];
            return_value[0][1] = -v[0];
            break;
          }

          case 3:
          {
            // In 3d, we can get two other vectors in a 3-step procedure:
            // - compute a 'w' that is definitely not collinear with 'v'
            // - compute the first direction as u[1] = v \times w,
            //   normalize it
            // - compute the second direction as u[2] = v \times u[1],
            //   normalize it
            //
            // For the first step, use a procedure suggested by Luca Heltai:
            // Set d to the index of the largest component of v. Make a vector
            // with this component equal to zero, and with the other two
            // equal to the norm of the point. Call this 'w'.
            unsigned int max_component = 0;
            for (unsigned int d=1; d<dim; ++d)
              if (std::fabs(v[d]) > std::fabs(v[max_component]))
                max_component = d;
            Tensor<1,dim> w = v;
            w[max_component] = 0;
            for (unsigned int d=1; d<dim; ++d)
              if (d != max_component)
                w[d] = v.norm();

            return_value[0] = cross_product_3d(v, w);
            return_value[0] *= v.norm() / return_value[0].norm();

            return_value[1] = cross_product_3d(v, return_value[0]);
            return_value[1] *= v.norm() / return_value[1].norm();

            break;
          }

          default:
            Assert (false, ExcNotImplemented());
        }

      return return_value;
    }


    //Evaluate the cosine and sine terms of a real spherical harmonic.
    //This is a fully normalized harmonic, that is to say, inner products
    //of these functions should integrate to a kronecker delta over
    //the surface of a sphere.
    std::pair<double,double> real_spherical_harmonic( const unsigned int l, //degree
                                                      const unsigned int m, //order
                                                      const double theta,   //colatitude (radians)
                                                      const double phi )    //longitude (radians)
    {
      const double sqrt_2 = numbers::SQRT2;
      const std::complex<double> sph_harm_val = boost::math::spherical_harmonic( l, m, theta, phi );
      if ( m == 0 )
        return std::make_pair( sph_harm_val.real(), 0.0 );
      else
        return std::make_pair( sqrt_2 * sph_harm_val.real(), sqrt_2 * sph_harm_val.imag() );
    }


    bool
    fexists(const std::string &filename)
    {
      std::ifstream ifile(filename.c_str());
      return static_cast<bool>(ifile); // only in c++11 you can convert to bool directly
    }



    std::string
    read_and_distribute_file_content(const std::string &filename,
                                     const MPI_Comm &comm)
    {
      std::string data_string;

      if (Utilities::MPI::this_mpi_process(comm) == 0)
        {
          std::ifstream filestream(filename.c_str());
          AssertThrow (filestream,
                       ExcMessage (std::string("Could not open file <") + filename + ">."));

          // Read data from disk
          std::stringstream datastream;
          filestream >> datastream.rdbuf();

          AssertThrow (filestream.eof(),
                       ExcMessage (std::string("Reading of file ") + filename + " finished " +
                                   "before the end of file was reached. Is the file corrupted or"
                                   "too large for the input buffer?"));

          data_string = datastream.str();
          unsigned int filesize = data_string.size();

          // Distribute data_size and data across processes
          MPI_Bcast(&filesize,1,MPI_UNSIGNED,0,comm);
          MPI_Bcast(&data_string[0],filesize,MPI_CHAR,0,comm);
        }
      else
        {
          // Prepare for receiving data
          unsigned int filesize;
          MPI_Bcast(&filesize,1,MPI_UNSIGNED,0,comm);
          data_string.resize(filesize);

          // Receive and store data
          MPI_Bcast(&data_string[0],filesize,MPI_CHAR,0,comm);
        }

      return data_string;
    }

    int
    mkdirp(std::string pathname,const mode_t mode)
    {
      // force trailing / so we can handle everything in loop
      if (pathname[pathname.size()-1] != '/')
        {
          pathname += '/';
        }

      size_t pre = 0;
      size_t pos;

      while ((pos = pathname.find_first_of('/',pre)) != std::string::npos)
        {
          const std::string subdir = pathname.substr(0,pos++);
          pre = pos;

          // if leading '/', first string is 0 length
          if (subdir.size() == 0)
            continue;

          int mkdir_return_value;
          if ((mkdir_return_value = mkdir(subdir.c_str(),mode)) && (errno != EEXIST))
            return mkdir_return_value;

        }

      return 0;
    }

    // tk does the cubic spline interpolation that can be used between different spherical layers in the mantle.
    // This interpolation is based on the script spline.h, which was downloaded from
    // http://kluge.in-chemnitz.de/opensource/spline/spline.h   //
    // Copyright (C) 2011, 2014 Tino Kluge (ttk448 at gmail.com)
    namespace tk
    {
      /**
       * Band matrix solver for a banded, square matrix of size @p dim. Used
       * by spline interpolation in tk::spline.
       */
      class band_matrix
      {
        public:
          /**
           * Constructor, see resize()
           */
          band_matrix(int dim, int n_u, int n_l);
          /**
           * Resize to a @p dim by @dim matrix with given number
           * of off-diagonals.
           */
          void resize(int dim, int n_u, int n_l);

          /**
           * Return the dimension of the matrix
           */
          int dim() const;

          /**
           * Number of off-diagonals above.
           */
          int num_upper() const
          {
            return m_upper.size()-1;
          }

          /**
           * Number of off-diagonals below.
           */
          int num_lower() const
          {
            return m_lower.size()-1;
          }

          /**
           * Writeable access to element A(i,j), indices going from
           * i=0,...,dim()-1
           */
          double &operator () (int i, int j);
          /**
           * Read-only access
           */
          double   operator () (int i, int j) const;

          /**
           * second diagonal (used in LU decomposition), saved in m_lower[0]
           */
          double &saved_diag(int i);

          /**
           * second diagonal (used in LU decomposition), saved in m_lower[0]
           */
          double  saved_diag(int i) const;

          /**
           * LU-Decomposition of a band matrix
           */
          void lu_decompose();

          /**
           * solves Ux=y
           */
          std::vector<double> r_solve(const std::vector<double> &b) const;

          /**
           * solves Ly=b
           */
          std::vector<double> l_solve(const std::vector<double> &b) const;

          /**
           * Solve Ax=b and builds LU decomposition using lu_decompose()
           * if @p is_lu_decomposed is false.
           */
          std::vector<double> lu_solve(const std::vector<double> &b,
                                       bool is_lu_decomposed=false);
        private:
          /**
           * diagonal and off-diagonals above
           */
          std::vector< std::vector<double> > m_upper;
          /**
           * diagonals below the diagonal
           */
          std::vector< std::vector<double> > m_lower;
      };

      band_matrix::band_matrix(int dim, int n_u, int n_l)
      {
        resize(dim, n_u, n_l);
      }

      void band_matrix::resize(int dim, int n_u, int n_l)
      {
        assert(dim>0);
        assert(n_u>=0);
        assert(n_l>=0);
        m_upper.resize(n_u+1);
        m_lower.resize(n_l+1);
        for (size_t i=0; i<m_upper.size(); i++)
          {
            m_upper[i].resize(dim);
          }
        for (size_t i=0; i<m_lower.size(); i++)
          {
            m_lower[i].resize(dim);
          }
      }

      int band_matrix::dim() const
      {
        if (m_upper.size()>0)
          {
            return m_upper[0].size();
          }
        else
          {
            return 0;
          }
      }

      double &band_matrix::operator () (int i, int j)
      {
        int k=j-i;       // what band is the entry
        assert( (i>=0) && (i<dim()) && (j>=0) && (j<dim()) );
        assert( (-num_lower()<=k) && (k<=num_upper()) );
        // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if (k>=0)
          return m_upper[k][i];
        else
          return m_lower[-k][i];
      }

      double band_matrix::operator () (int i, int j) const
      {
        int k=j-i;       // what band is the entry
        assert( (i>=0) && (i<dim()) && (j>=0) && (j<dim()) );
        assert( (-num_lower()<=k) && (k<=num_upper()) );
        // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if (k>=0)
          return m_upper[k][i];
        else
          return m_lower[-k][i];
      }

      double band_matrix::saved_diag(int i) const
      {
        assert( (i>=0) && (i<dim()) );
        return m_lower[0][i];
      }

      double &band_matrix::saved_diag(int i)
      {
        assert( (i>=0) && (i<dim()) );
        return m_lower[0][i];
      }

      void band_matrix::lu_decompose()
      {
        int  i_max,j_max;
        int  j_min;
        double x;

        // preconditioning
        //             // normalize column i so that a_ii=1
        for (int i=0; i<this->dim(); i++)
          {
            assert(this->operator()(i,i)!=0.0);
            this->saved_diag(i)=1.0/this->operator()(i,i);
            j_min=std::max(0,i-this->num_lower());
            j_max=std::min(this->dim()-1,i+this->num_upper());
            for (int j=j_min; j<=j_max; j++)
              {
                this->operator()(i,j) *= this->saved_diag(i);
              }
            this->operator()(i,i)=1.0;          // prevents rounding errors
          }

        // Gauss LR-Decomposition
        for (int k=0; k<this->dim(); k++)
          {
            i_max=std::min(this->dim()-1,k+this->num_lower());  // num_lower not a mistake!
            for (int i=k+1; i<=i_max; i++)
              {
                assert(this->operator()(k,k)!=0.0);
                x=-this->operator()(i,k)/this->operator()(k,k);
                this->operator()(i,k)=-x;                         // assembly part of L
                j_max=std::min(this->dim()-1,k+this->num_upper());
                for (int j=k+1; j<=j_max; j++)
                  {
                    // assembly part of R
                    this->operator()(i,j)=this->operator()(i,j)+x*this->operator()(k,j);
                  }
              }
          }
      }

      std::vector<double> band_matrix::l_solve(const std::vector<double> &b) const
      {
        assert( this->dim()==(int)b.size() );
        std::vector<double> x(this->dim());
        int j_start;
        double sum;
        for (int i=0; i<this->dim(); i++)
          {
            sum=0;
            j_start=std::max(0,i-this->num_lower());
            for (int j=j_start; j<i; j++) sum += this->operator()(i,j)*x[j];
            x[i]=(b[i]*this->saved_diag(i)) - sum;
          }
        return x;
      }


      std::vector<double> band_matrix::r_solve(const std::vector<double> &b) const
      {
        assert( this->dim()==(int)b.size() );
        std::vector<double> x(this->dim());
        int j_stop;
        double sum;
        for (int i=this->dim()-1; i>=0; i--)
          {
            sum=0;
            j_stop=std::min(this->dim()-1,i+this->num_upper());
            for (int j=i+1; j<=j_stop; j++) sum += this->operator()(i,j)*x[j];
            x[i]=( b[i] - sum ) / this->operator()(i,i);
          }
        return x;
      }

      std::vector<double> band_matrix::lu_solve(const std::vector<double> &b,
                                                bool is_lu_decomposed)
      {
        assert( this->dim()==(int)b.size() );
        std::vector<double>  x,y;
        // TODO: this is completely unsafe because you rely on the user
        // if the function is called more than once.
        if (is_lu_decomposed==false)
          {
            this->lu_decompose();
          }
        y=this->l_solve(b);
        x=this->r_solve(y);
        return x;
      }


      void spline::set_points(const std::vector<double> &x,
                              const std::vector<double> &y,
                              bool cubic_spline)
      {
        assert(x.size()==y.size());
        m_x=x;
        m_y=y;
        int   n=x.size();
        for (int i=0; i<n-1; i++)
          {
            assert(m_x[i]<m_x[i+1]);
          }

        if (cubic_spline==true)  // cubic spline interpolation
          {
            // setting up the matrix and right hand side of the equation system
            // for the parameters b[]
            band_matrix A(n,1,1);
            std::vector<double>  rhs(n);
            for (int i=1; i<n-1; i++)
              {
                A(i,i-1)=1.0/3.0*(x[i]-x[i-1]);
                A(i,i)=2.0/3.0*(x[i+1]-x[i-1]);
                A(i,i+1)=1.0/3.0*(x[i+1]-x[i]);
                rhs[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
              }
            // boundary conditions, zero curvature b[0]=b[n-1]=0
            A(0,0)=2.0;
            A(0,1)=0.0;
            rhs[0]=0.0;
            A(n-1,n-1)=2.0;
            A(n-1,n-2)=0.0;
            rhs[n-1]=0.0;

            // solve the equation system to obtain the parameters b[]
            m_b=A.lu_solve(rhs);

            // calculate parameters a[] and c[] based on b[]
            m_a.resize(n);
            m_c.resize(n);
            for (int i=0; i<n-1; i++)
              {
                m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/(x[i+1]-x[i]);
                m_c[i]=(y[i+1]-y[i])/(x[i+1]-x[i])
                       - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*(x[i+1]-x[i]);
              }
          }
        else     // linear interpolation
          {
            m_a.resize(n);
            m_b.resize(n);
            m_c.resize(n);
            for (int i=0; i<n-1; i++)
              {
                m_a[i]=0.0;
                m_b[i]=0.0;
                m_c[i]=(m_y[i+1]-m_y[i])/(m_x[i+1]-m_x[i]);
              }
          }

        // for the right boundary we define
        // f_{n-1}(x) = b*(x-x_{n-1})^2 + c*(x-x_{n-1}) + y_{n-1}
        double h=x[n-1]-x[n-2];
        // m_b[n-1] is determined by the boundary condition
        m_a[n-1]=0.0;
        m_c[n-1]=3.0*m_a[n-2]*h*h+2.0*m_b[n-2]*h+m_c[n-2];   // = f'_{n-2}(x_{n-1})
      }

      double spline::operator() (double x) const
      {
        size_t n=m_x.size();
        // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
        std::vector<double>::const_iterator it;
        it=std::lower_bound(m_x.begin(),m_x.end(),x);
        int idx=std::max( int(it-m_x.begin())-1, 0);

        double h=x-m_x[idx];
        double interpol;
        if (x<m_x[0])
          {
            // extrapolation to the left
            interpol=((m_b[0])*h + m_c[0])*h + m_y[0];
          }
        else if (x>m_x[n-1])
          {
            // extrapolation to the right
            interpol=((m_b[n-1])*h + m_c[n-1])*h + m_y[n-1];
          }
        else
          {
            // interpolation
            interpol=((m_a[idx]*h + m_b[idx])*h + m_c[idx])*h + m_y[idx];
          }
        return interpol;
      }

    } // namespace tk



    template <int dim>
    AsciiDataLookup<dim>::AsciiDataLookup(const unsigned int components,
                                          const double scale_factor)
      :
      components(components),
      data(components),
      scale_factor(scale_factor)
    {}

    template <int dim>
    void
    AsciiDataLookup<dim>::load_file(const std::string &filename,
                                    const MPI_Comm &comm)
    {
      // Read data from disk and distribute among processes
      std::stringstream in(read_and_distribute_file_content(filename, comm));

      // Read header lines and table size
      while (in.peek() == '#')
        {
          std::string line;
          getline(in,line);
          std::stringstream linestream(line);
          std::string word;
          while (linestream >> word)
            if (word == "POINTS:")
              for (unsigned int i = 0; i < dim; i++)
                {
                  unsigned int temp_index;
                  linestream >> temp_index;

                  if (table_points[i] == 0)
                    table_points[i] = temp_index;
                  else
                    AssertThrow (table_points[i] == temp_index,
                                 ExcMessage("The file grid must not change over model runtime. "
                                            "Either you prescribed a conflicting number of points in "
                                            "the input file, or the POINTS comment in your data files "
                                            "is changing between following files."));
                }
        }

      for (unsigned int i = 0; i < dim; i++)
        {
          AssertThrow(table_points[i] != 0,
                      ExcMessage("Could not successfully read in the file header of the "
                                 "ascii data file <" + filename + ">. One header line has to "
                                 "be of the format: '#POINTS: N1 [N2] [N3]', where N1 and "
                                 "potentially N2 and N3 have to be the number of data points "
                                 "in their respective dimension. Check for typos in this line "
                                 "(e.g. a missing space character)."));
        }

      /**
       * Table for the new data. This peculiar reinit is necessary, because
       * there is no constructor for Table, which takes TableIndices as
       * argument.
       */
      Table<dim,double> data_table;
      data_table.TableBase<dim,double>::reinit(table_points);
      std::vector<Table<dim,double> > data_tables(components+dim,data_table);

      // Read data lines
      unsigned int i = 0;
      double temp_data;

      while (in >> temp_data)
        {
          const unsigned int column_num = i%(components+dim);

          if (column_num >= dim)
            temp_data *= scale_factor;

          data_tables[column_num](compute_table_indices(i)) = temp_data;

          i++;

          // TODO: add checks for coordinate ordering in data files
        }

      AssertThrow(i == (components + dim) * data_table.n_elements(),
                  ExcMessage (std::string("Number of read in points does not match number of expected points. File corrupted?")));

      std_cxx11::array<unsigned int,dim> table_intervals;

      for (unsigned int i = 0; i < dim; i++)
        {
          table_intervals[i] = table_points[i] - 1;

          TableIndices<dim> idx;
          grid_extent[i].first = data_tables[i](idx);
          idx[i] = table_points[i] - 1;
          grid_extent[i].second = data_tables[i](idx);
        }

      for (unsigned int i = 0; i < components; i++)
        {
          if (data[i])
            delete data[i];
          data[i] = new Functions::InterpolatedUniformGridData<dim> (grid_extent,
                                                                     table_intervals,
                                                                     data_tables[dim+i]);
        }
    }


    template <int dim>
    double
    AsciiDataLookup<dim>::get_data(const Point<dim> &position,
                                   const unsigned int component) const
    {
      return data[component]->value(position);
    }


    template <int dim>
    TableIndices<dim>
    AsciiDataLookup<dim>::compute_table_indices(const unsigned int i) const
    {
      TableIndices<dim> idx;
      idx[0] = (i / (components+dim)) % table_points[0];
      if (dim >= 2)
        idx[1] = ((i / (components+dim)) / table_points[0]) % table_points[1];
      if (dim == 3)
        idx[2] = (i / (components+dim)) / (table_points[0] * table_points[1]);

      return idx;
    }



    template <int dim>
    AsciiDataBase<dim>::AsciiDataBase ()
    {}


    template <int dim>
    void
    AsciiDataBase<dim>::declare_parameters (ParameterHandler  &prm,
                                            const std::string &default_directory,
                                            const std::string &default_filename)
    {
      prm.enter_subsection ("Ascii data model");
      {
        prm.declare_entry ("Data directory",
                           default_directory,
                           Patterns::DirectoryName (),
                           "The name of a directory that contains the model data. This path "
                           "may either be absolute (if starting with a '/') or relative to "
                           "the current directory. The path may also include the special "
                           "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                           "in which the ASPECT source files were located when ASPECT was "
                           "compiled. This interpretation allows, for example, to reference "
                           "files located in the 'data/' subdirectory of ASPECT. ");
        prm.declare_entry ("Data file name",
                           default_filename,
                           Patterns::Anything (),
                           "The file name of the material data. Provide file in format: "
                           "(Velocity file name).\\%s\\%d where \\%s is a string specifying "
                           "the boundary of the model according to the names of the boundary "
                           "indicators (of a box or a spherical shell).\\%d is any sprintf integer "
                           "qualifier, specifying the format of the current file number. ");
        prm.declare_entry ("Scale factor", "1",
                           Patterns::Double (0),
                           "Scalar factor, which is applied to the boundary velocity. "
                           "You might want to use this to scale the velocities to a "
                           "reference model (e.g. with free-slip boundary) or another "
                           "plate reconstruction. Another way to use this factor is to "
                           "convert units of the input files. The unit is assumed to be"
                           "m/s or m/yr depending on the 'Use years in output instead of "
                           "seconds' flag. If you provide velocities in cm/yr set this "
                           "factor to 0.01.");
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    AsciiDataBase<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Ascii data model");
      {
        // Get the path to the data files. If it contains a reference
        // to $ASPECT_SOURCE_DIR, replace it by what CMake has given us
        // as a #define
        data_directory = Utilities::replace_in_string(prm.get ("Data directory"),
                                                      "$ASPECT_SOURCE_DIR",
                                                      ASPECT_SOURCE_DIR);
        data_file_name    = prm.get ("Data file name");
        scale_factor      = prm.get_double ("Scale factor");
      }
      prm.leave_subsection();
    }

    template <int dim>
    AsciiDataBoundary<dim>::AsciiDataBoundary ()
      :
      current_file_number(0),
      first_data_file_model_time(0.0),
      first_data_file_number(0),
      decreasing_file_order(false),
      data_file_time_step(0.0),
      time_weight(0.0),
      time_dependent(true),
      lookups(),
      old_lookups()
    {}

    template <int dim>
    void
    AsciiDataBoundary<dim>::initialize(const std::set<types::boundary_id> &boundary_ids,
                                       const unsigned int components)
    {
      AssertThrow ((dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()))
                   || (dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model())) != 0
                   || (dynamic_cast<const GeometryModel::Box<dim>*> (&this->get_geometry_model())) != 0,
                   ExcMessage ("This ascii data plugin can only be used when using "
                               "a spherical shell, chunk or box geometry."));


      for (typename std::set<types::boundary_id>::const_iterator
           boundary_id = boundary_ids.begin();
           boundary_id != boundary_ids.end(); ++boundary_id)
        {
          std_cxx11::shared_ptr<Utilities::AsciiDataLookup<dim-1> > lookup;
          lookup.reset(new Utilities::AsciiDataLookup<dim-1> (components,
                                                              Utilities::AsciiDataBase<dim>::scale_factor));
          lookups.insert(std::make_pair(*boundary_id,lookup));

          lookup.reset(new Utilities::AsciiDataLookup<dim-1> (components,
                                                              Utilities::AsciiDataBase<dim>::scale_factor));
          old_lookups.insert(std::make_pair(*boundary_id,lookup));


          // Set the first file number and load the first files
          current_file_number = first_data_file_number;

          const int next_file_number =
            (decreasing_file_order) ?
            current_file_number - 1
            :
            current_file_number + 1;

          this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
                            << create_filename (current_file_number,*boundary_id) << "." << std::endl << std::endl;

          const std::string filename (create_filename (current_file_number,*boundary_id));
          if (fexists(filename))
            lookups.find(*boundary_id)->second->load_file(filename,this->get_mpi_communicator());
          else
            AssertThrow(false,
                        ExcMessage (std::string("Ascii data file <")
                                    +
                                    filename
                                    +
                                    "> not found!"));

          // If the boundary condition is constant, switch off time_dependence
          // immediately. If not, also load the second file for interpolation.
          // This catches the case that many files are present, but the
          // parameter file requests a single file.
          if (create_filename (current_file_number,*boundary_id) == create_filename (current_file_number+1,*boundary_id))
            {
              end_time_dependence ();
            }
          else
            {
              const std::string filename (create_filename (next_file_number,*boundary_id));
              this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
                                << filename << "." << std::endl << std::endl;
              if (fexists(filename))
                {
                  lookups.find(*boundary_id)->second.swap(old_lookups.find(*boundary_id)->second);
                  lookups.find(*boundary_id)->second->load_file(filename,this->get_mpi_communicator());
                }
              else
                end_time_dependence ();
            }
        }
    }


    template <int dim>
    std_cxx11::array<unsigned int,dim-1>
    AsciiDataBoundary<dim>::get_boundary_dimensions (const types::boundary_id boundary_id) const
    {
      std_cxx11::array<unsigned int,dim-1> boundary_dimensions;

      switch (dim)
        {
          case 2:
            if ((boundary_id == 2) || (boundary_id == 3))
              {
                boundary_dimensions[0] = 0;
              }
            else if ((boundary_id == 0) || (boundary_id == 1))
              {
                boundary_dimensions[0] = 1;
              }
            else
              {
                boundary_dimensions[0] = numbers::invalid_unsigned_int;
                AssertThrow(false,ExcNotImplemented());
              }

            break;

          case 3:
            if ((boundary_id == 4) || (boundary_id == 5))
              {
                boundary_dimensions[0] = 0;
                boundary_dimensions[1] = 1;
              }
            else if ((boundary_id == 0) || (boundary_id == 1))
              {
                boundary_dimensions[0] = 1;
                boundary_dimensions[1] = 2;
              }
            else if ((boundary_id == 2) || (boundary_id == 3))
              {
                boundary_dimensions[0] = 0;
                boundary_dimensions[1] = 2;
              }
            else
              {
                boundary_dimensions[0] = numbers::invalid_unsigned_int;
                boundary_dimensions[1] = numbers::invalid_unsigned_int;
                AssertThrow(false,ExcNotImplemented());
              }

            break;

          default:
            for (unsigned int d=0; d<dim-1; ++d)
              boundary_dimensions[d] = numbers::invalid_unsigned_int;
            AssertThrow(false,ExcNotImplemented());
        }
      return boundary_dimensions;
    }

    template <int dim>
    std::string
    AsciiDataBoundary<dim>::create_filename (const int filenumber,
                                             const types::boundary_id boundary_id) const
    {
      std::string templ = Utilities::AsciiDataBase<dim>::data_directory + Utilities::AsciiDataBase<dim>::data_file_name;
      const int size = templ.length();
      const std::string boundary_name = this->get_geometry_model().translate_id_to_symbol_name(boundary_id);
      char *filename = (char *) (malloc ((size + 10) * sizeof(char)));
      snprintf (filename, size + 10, templ.c_str (), boundary_name.c_str(),filenumber);
      std::string str_filename (filename);
      free (filename);
      return str_filename;
    }


    template <int dim>
    void
    AsciiDataBoundary<dim>::update ()
    {
      if (time_dependent && (this->get_time() - first_data_file_model_time >= 0.0))
        {
          const double time_steps_since_start = (this->get_time() - first_data_file_model_time)
                                                / data_file_time_step;
          // whether we need to update our data files. This looks so complicated
          // because we need to catch increasing and decreasing file orders and all
          // possible first_data_file_model_times and first_data_file_numbers.
          const bool need_update =
            static_cast<int> (time_steps_since_start)
            > std::abs(current_file_number - first_data_file_number);

          if (need_update)
            {
              // The last file, which was tried to be loaded was
              // number current_file_number +/- 1, because current_file_number
              // is the file older than the current model time
              const int old_file_number =
                (decreasing_file_order) ?
                current_file_number - 1
                :
                current_file_number + 1;

              //Calculate new file_number
              current_file_number =
                (decreasing_file_order) ?
                first_data_file_number
                - static_cast<unsigned int> (time_steps_since_start)
                :
                first_data_file_number
                + static_cast<unsigned int> (time_steps_since_start);

              const bool load_both_files = std::abs(current_file_number - old_file_number) >= 1;

              for (typename std::map<types::boundary_id,
                   std_cxx11::shared_ptr<Utilities::AsciiDataLookup<dim-1> > >::iterator
                   boundary_id = lookups.begin();
                   boundary_id != lookups.end(); ++boundary_id)
                update_data(boundary_id->first,load_both_files);
            }

          time_weight = time_steps_since_start
                        - std::abs(current_file_number - first_data_file_number);

          Assert ((0 <= time_weight) && (time_weight <= 1),
                  ExcMessage (
                    "Error in set_current_time. Time_weight has to be in [0,1]"));
        }
    }

    template <int dim>
    void
    AsciiDataBoundary<dim>::update_data (const types::boundary_id boundary_id,
                                         const bool load_both_files)
    {
      // If the time step was large enough to move forward more
      // then one data file we need to load both current files
      // to stay accurate in interpolation
      if (load_both_files)
        {
          const std::string filename (create_filename (current_file_number,boundary_id));
          this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
                            << filename << "." << std::endl << std::endl;
          if (fexists(filename))
            {
              lookups.find(boundary_id)->second.swap(old_lookups.find(boundary_id)->second);
              lookups.find(boundary_id)->second->load_file(filename,this->get_mpi_communicator());
            }

          // If loading current_time_step failed, end time dependent part with old_file_number.
          else
            end_time_dependence ();
        }

      // Now load the next data file. This part is the main purpose of this function.
      const int next_file_number =
        (decreasing_file_order) ?
        current_file_number - 1
        :
        current_file_number + 1;

      const std::string filename (create_filename (next_file_number,boundary_id));
      this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
                        << filename << "." << std::endl << std::endl;
      if (fexists(filename))
        {
          lookups.find(boundary_id)->second.swap(old_lookups.find(boundary_id)->second);
          lookups.find(boundary_id)->second->load_file(filename,this->get_mpi_communicator());
        }

      // If next file does not exist, end time dependent part with current_time_step.
      else
        end_time_dependence ();
    }

    template <int dim>
    void
    AsciiDataBoundary<dim>::end_time_dependence ()
    {
      // no longer consider the problem time dependent from here on out
      // this cancels all attempts to read files at the next time steps
      time_dependent = false;
      // Give warning if first processor
      this->get_pcout() << std::endl
                        << "   Loading new data file did not succeed." << std::endl
                        << "   Assuming constant boundary conditions for rest of model run."
                        << std::endl << std::endl;
    }

    template <int dim>
    double
    AsciiDataBoundary<dim>::
    get_data_component (const types::boundary_id             boundary_indicator,
                        const Point<dim>                    &position,
                        const unsigned int                   component) const
    {
      if (this->get_time() - first_data_file_model_time >= 0.0)
        {
          Point<dim> internal_position = position;

          if (dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()) != 0)
            {
              const std_cxx11::array<double,dim> spherical_position =
                ::aspect::Utilities::spherical_coordinates(position);

              for (unsigned int i = 0; i < dim; i++)
                internal_position[i] = spherical_position[i];
            }

          const std_cxx11::array<unsigned int,dim-1> boundary_dimensions =
            get_boundary_dimensions(boundary_indicator);

          Point<dim-1> data_position;
          for (unsigned int i = 0; i < dim-1; i++)
            data_position[i] = internal_position[boundary_dimensions[i]];

          const double data = lookups.find(boundary_indicator)->second->get_data(data_position,component);

          if (!time_dependent)
            return data;

          const double old_data = old_lookups.find(boundary_indicator)->second->get_data(data_position,component);

          return time_weight * data + (1 - time_weight) * old_data;
        }
      else
        return 0.0;
    }


    template <int dim>
    void
    AsciiDataBoundary<dim>::declare_parameters (ParameterHandler  &prm,
                                                const std::string &default_directory,
                                                const std::string &default_filename)
    {
      Utilities::AsciiDataBase<dim>::declare_parameters(prm,
                                                        default_directory,
                                                        default_filename);

      prm.enter_subsection ("Ascii data model");
      {
        prm.declare_entry ("Data file time step", "1e6",
                           Patterns::Double (0),
                           "Time step between following velocity files. "
                           "Depending on the setting of the global 'Use years in output instead of seconds' flag "
                           "in the input file, this number is either interpreted as seconds or as years. "
                           "The default is one million, i.e., either one million seconds or one million years.");
        prm.declare_entry ("First data file model time", "0",
                           Patterns::Double (0),
                           "Time from which on the velocity file with number 'First velocity "
                           "file number' is used as boundary condition. Previous to this "
                           "time, a no-slip boundary condition is assumed. Depending on the setting "
                           "of the global 'Use years in output instead of seconds' flag "
                           "in the input file, this number is either interpreted as seconds or as years.");
        prm.declare_entry ("First data file number", "0",
                           Patterns::Integer (),
                           "Number of the first velocity file to be loaded when the model time "
                           "is larger than 'First velocity file model time'.");
        prm.declare_entry ("Decreasing file order", "false",
                           Patterns::Bool (),
                           "In some cases the boundary files are not numbered in increasing "
                           "but in decreasing order (e.g. 'Ma BP'). If this flag is set to "
                           "'True' the plugin will first load the file with the number "
                           "'First velocity file number' and decrease the file number during "
                           "the model run.");
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    AsciiDataBoundary<dim>::parse_parameters (ParameterHandler &prm)
    {
      Utilities::AsciiDataBase<dim>::parse_parameters(prm);

      prm.enter_subsection("Ascii data model");
      {
        data_file_time_step             = prm.get_double ("Data file time step");
        first_data_file_model_time      = prm.get_double ("First data file model time");
        first_data_file_number          = prm.get_double ("First data file number");
        decreasing_file_order           = prm.get_bool   ("Decreasing file order");

        if (this->convert_output_to_years() == true)
          {
            data_file_time_step        *= year_in_seconds;
            first_data_file_model_time *= year_in_seconds;
          }
      }
      prm.leave_subsection();
    }

    template <int dim>
    AsciiDataInitial<dim>::AsciiDataInitial ()
    {}


    template <int dim>
    void
    AsciiDataInitial<dim>::initialize (const unsigned int components)
    {
      AssertThrow ((dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()))
                   || (dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model())) != 0
                   || (dynamic_cast<const GeometryModel::Box<dim>*> (&this->get_geometry_model())) != 0,
                   ExcMessage ("This ascii data plugin can only be used when using "
                               "a spherical shell, chunk or box geometry."));

      lookup.reset(new Utilities::AsciiDataLookup<dim> (components,
                                                        Utilities::AsciiDataBase<dim>::scale_factor));

      const std::string filename = Utilities::AsciiDataBase<dim>::data_directory
                                   + Utilities::AsciiDataBase<dim>::data_file_name;

      this->get_pcout() << std::endl << "   Loading Ascii data initial file "
                        << filename << "." << std::endl << std::endl;

      if (fexists(filename))
        lookup->load_file(filename,this->get_mpi_communicator());
      else
        AssertThrow(false,
                    ExcMessage (std::string("Ascii data file <")
                                +
                                filename
                                +
                                "> not found!"));
    }

    template <int dim>
    double
    AsciiDataInitial<dim>::
    get_data_component (const Point<dim>                    &position,
                        const unsigned int                   component) const
    {
      Point<dim> internal_position = position;

      if (dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()) != 0
          || (dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model())) != 0)
        {
          const std_cxx11::array<double,dim> spherical_position =
            ::aspect::Utilities::spherical_coordinates(position);

          for (unsigned int i = 0; i < dim; i++)
            internal_position[i] = spherical_position[i];
        }
      return lookup->get_data(internal_position,component);
    }

    // Explicit instantiations
    template class AsciiDataLookup<1>;
    template class AsciiDataLookup<2>;
    template class AsciiDataLookup<3>;
    template class AsciiDataBase<2>;
    template class AsciiDataBase<3>;
    template class AsciiDataBoundary<2>;
    template class AsciiDataBoundary<3>;
    template class AsciiDataInitial<2>;
    template class AsciiDataInitial<3>;

    template Point<2> cartesian_coordinates<2>(const std_cxx11::array<double,2> &scoord);
    template Point<3> cartesian_coordinates<3>(const std_cxx11::array<double,3> &scoord);

    template std_cxx11::array<double,2> spherical_coordinates<2>(const Point<2> &position);
    template std_cxx11::array<double,3> spherical_coordinates<3>(const Point<3> &position);

    template std_cxx11::array<Tensor<1,2>,1> orthogonal_vectors (const Tensor<1,2> &v);
    template std_cxx11::array<Tensor<1,3>,2> orthogonal_vectors (const Tensor<1,3> &v);
  }
}
