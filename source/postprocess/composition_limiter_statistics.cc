/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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


#include <aspect/postprocess/composition_BPL_statistics.h>
#include <aspect/postprocess/composition_statistics.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    CompositionBPLimiterStatistics<dim>::execute (TableHandler &statistics)
    {
      if (this->n_compositional_fields() == 0)
        return std::pair<std::string,std::string>();

      // create a quadrature formula based on the compositional element alone.
      // be defensive about determining that a compositional field actually exists
      AssertThrow (this->introspection().base_elements.compositional_fields
                   != numbers::invalid_unsigned_int,
                   ExcMessage("This postprocessor cannot be used without compositional fields."));
      const QGauss<dim> quadrature_formula (this->get_fe().base_element(this->introspection().base_elements.compositional_fields).degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_quadrature_points |
                               update_JxW_values);

      std::vector<double> compositional_values(n_q_points);

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();

      std::vector<double> local_compositional_integrals (this->n_compositional_fields());
      std::vector<double> local_compositional_average_integrals (this->n_compositional_fields());
      double local_area_integrals;
      double C_max=1.0;
      double C_min=0;
      const unsigned int dofs_per_cell   = this->get_fe().dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      // compute the integral quantities by quadrature
      for (; cell!=endc; ++cell)
      {
        cell->get_dof_indices (local_dof_indices);
        double min_composition, max_composition;
        local_area_integrals=0;
        if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            local_area_integrals+= fe_values.JxW(q);
          }
          std::cout << "local average"<< local_area_integrals<<std::endl;
          for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
          {
            local_compositional_average_integrals[c]=0;
            fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                compositional_values);
            for (unsigned int q=0; q<n_q_points; ++q)
            {
              local_compositional_average_integrals[c]+= compositional_values[q]*fe_values.JxW(q);
            }
            local_compositional_average_integrals[c] /= local_area_integrals;
            std::cout << "local compositional average"<< local_compositional_average_integrals[c]<<std::endl;

            min_composition=compositional_values[0];
            max_composition=compositional_values[0];
            for (unsigned int q=0; q<n_q_points; ++q)
            {
              min_composition = std::min<double> (min_composition,
                  compositional_values[q]);
              max_composition = std::max<double> (max_composition,
                  compositional_values[q]);
            }
            std::cout << "local max/min"<< min_composition<<max_composition<<std::endl;
            // find the trouble cell
            if (min_composition <C_min || max_composition >C_max)
            {
              //Define theta
              double theta_T=std::min<double>(1,abs((C_max-local_compositional_average_integrals[c])
                    /(max_composition-local_compositional_average_integrals[c])));
              theta_T=std::min<double>(theta_T,abs((C_min-local_compositional_average_integrals[c])
                    /(min_composition-local_compositional_average_integrals[c])));
              std::cout << "theta_T"<< theta_T<<std::endl;
              //Modify the numerical solution at freedoms
              IndexSet range = this->get_solution().block(this->introspection().block_indices.compositional_fields[c]).locally_owned_elements();
              std::vector<double> t_tmp(range.n_elements());
              for (unsigned int i=0; i<range.n_elements(); ++i)
              {
                const unsigned int idx = range.nth_index_in_set(i);
                t_tmp[i]=this->get_solution().block(this->introspection().block_indices.compositional_fields[c])(idx);
                std::cout << "C before "<< t_tmp[i]<<std::endl;
                t_tmp[i]=theta_T*(t_tmp[i]-local_compositional_average_integrals[c])+local_compositional_average_integrals[c];
                std::cout << "C after"<< t_tmp[i]<<std::endl;
              }
            min_composition=t_tmp[0];
            max_composition=t_tmp[0];
            for (unsigned int q=0; q<range.n_elements(); ++q)
            {
              min_composition = std::min<double> (min_composition,
                  t_tmp[q]);
              max_composition = std::max<double> (max_composition,
                  t_tmp[q]);
            }
            std::cout << "local max/min after limiter"<< min_composition<<max_composition<<std::endl;
            }


          }


        }


      }
      // compute the sum over all processors
      std::vector<double> global_compositional_integrals (local_compositional_integrals.size());
      Utilities::MPI::sum (local_compositional_integrals,
          this->get_mpi_communicator(),
          global_compositional_integrals);

      // compute min/max by simply
      // looping over the elements of the
      // solution vector.
      std::vector<double> local_min_compositions (this->n_compositional_fields(),
          std::numeric_limits<double>::max());
      std::vector<double> local_max_compositions (this->n_compositional_fields(),
          -std::numeric_limits<double>::max());

      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
      {
        IndexSet range = this->get_solution().block(this->introspection().block_indices.compositional_fields[c]).locally_owned_elements();
        for (unsigned int i=0; i<range.n_elements(); ++i)
        {
          const unsigned int idx = range.nth_index_in_set(i);
          const double val =  this->get_solution().block(this->introspection().block_indices.compositional_fields[c])(idx);

          local_min_compositions[c] = std::min<double> (local_min_compositions[c], val);
          local_max_compositions[c] = std::max<double> (local_max_compositions[c], val);
        }

      }
      // now do the reductions over all processors
      std::vector<double> global_min_compositions (this->n_compositional_fields(),
          std::numeric_limits<double>::max());
      std::vector<double> global_max_compositions (this->n_compositional_fields(),
          -std::numeric_limits<double>::max());
      {
        Utilities::MPI::min (local_min_compositions,
            this->get_mpi_communicator(),
            global_min_compositions);
        Utilities::MPI::max (local_max_compositions,
            this->get_mpi_communicator(),
            global_max_compositions);
      }

      // finally produce something for the statistics file
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
      {
        statistics.add_value ("Minimal value for composition " + this->introspection().name_for_compositional_index(c),
            global_min_compositions[c]);
        statistics.add_value ("Maximal value for composition " + this->introspection().name_for_compositional_index(c),
            global_max_compositions[c]);
        statistics.add_value ("Global mass for composition " + this->introspection().name_for_compositional_index(c),
            global_compositional_integrals[c]);
      }

      // also make sure that the other columns filled by the this object
      // all show up with sufficient accuracy and in scientific notation
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
      {
        const std::string columns[] = { "Minimal value for composition " + this->introspection().name_for_compositional_index(c),
          "Maximal value for composition " + this->introspection().name_for_compositional_index(c),
          "Global mass for composition " + this->introspection().name_for_compositional_index(c)
        };
        for (unsigned int i=0; i<sizeof(columns)/sizeof(columns[0]); ++i)
        {
          statistics.set_precision (columns[i], 8);
          statistics.set_scientific (columns[i], true);
        }
      }

      std::ostringstream output;
      output.precision(4);
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
      {
        output << global_min_compositions[c] << '/'
          << global_max_compositions[c] << '/'
          << global_compositional_integrals[c];
        if (c+1 != this->n_compositional_fields())
          output << " // ";
      }

      return std::pair<std::string, std::string> ("Compositions min/max/mass:",
          output.str());
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(CompositionBPLimiterStatistics,
        "composition BPL statistics",
        "A postprocessor that computes some statistics about "
        "the compositional fields, if present in this simulation. "
        "In particular, it computes maximal and minimal values of "
        "each field, as well as the total mass contained in this "
        "field as defined by the integral "
        "$m_i(t) = \\int_\\Omega c_i(\\mathbf x,t) \\; dx$.")
  }
}
