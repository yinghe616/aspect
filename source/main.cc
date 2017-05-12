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


#include <aspect/simulator.h>
#include <aspect/utilities.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>

#include <string>

#ifdef DEBUG
#ifdef ASPECT_USE_FP_EXCEPTIONS
#include <fenv.h>
#endif
#endif

#if ASPECT_USE_SHARED_LIBS==1
#  include <dlfcn.h>
#  ifdef ASPECT_HAVE_LINK_H
#    include <link.h>
#  endif
#endif



// get the value of a particular parameter from the contents of the input
// file. return an empty string if not found
std::string
get_last_value_of_parameter(const std::string &parameters,
                            const std::string &parameter_name)
{
  std::string return_value;

  std::istringstream x_file(parameters);
  while (x_file)
    {
      // get one line and strip spaces at the front and back
      std::string line;
      std::getline(x_file, line);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      while ((line.size() > 0)
             && (line[line.size() - 1] == ' ' || line[line.size() - 1] == '\t'))
        line.erase(line.size() - 1, std::string::npos);
      // now see whether the line starts with 'set' followed by multiple spaces
      // if not, try next line
      if (line.size() < 4)
        continue;

      if ((line[0] != 's') || (line[1] != 'e') || (line[2] != 't')
          || !(line[3] == ' ' || line[3] == '\t'))
        continue;

      // delete the "set " and then delete more spaces if present
      line.erase(0, 4);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      // now see whether the next word is the word we look for
      if (line.find(parameter_name) != 0)
        continue;

      line.erase(0, parameter_name.size());
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);

      // we'd expect an equals size here
      if ((line.size() < 1) || (line[0] != '='))
        continue;

      // remove comment
      std::string::size_type pos = line.find('#');
      if (pos != std::string::npos)
        line.erase (pos);

      // trim the equals sign at the beginning and possibly following spaces
      // as well as spaces at the end
      line.erase(0, 1);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      while ((line.size() > 0) && (line[line.size()-1] == ' ' || line[line.size()-1] == '\t'))
        line.erase(line.size()-1, std::string::npos);

      // the rest should now be what we were looking for
      return_value = line;
    }

  return return_value;
}


// extract the dimension in which to run ASPECT from the
// the contents of the parameter file. this is something that
// we need to do before processing the parameter file since we
// need to know whether to use the dim=2 or dim=3 instantiation
// of the main classes
unsigned int
get_dimension(const std::string &parameters)
{
  const std::string dimension = get_last_value_of_parameter(parameters, "Dimension");
  if (dimension.size() > 0)
    return dealii::Utilities::string_to_int (dimension);
  else
    return 2;
}


#if ASPECT_USE_SHARED_LIBS==1

#ifdef ASPECT_HAVE_LINK_H
// collect the names of the shared libraries linked to by this program. this
// function is a callback for the dl_iterate_phdr() function we call below
int get_names_of_shared_libs (struct dl_phdr_info *info,
                              size_t,
                              void *data)
{
  reinterpret_cast<std::set<std::string>*>(data)->insert (info->dlpi_name);
  return 0;
}
#endif


// make sure the list of shared libraries we currently link with
// has deal.II only once
void validate_shared_lib_list (const bool before_loading_shared_libs)
{
#ifdef ASPECT_HAVE_LINK_H
  // get the list of all shared libs we currently link against
  std::set<std::string> shared_lib_names;
  dl_iterate_phdr(get_names_of_shared_libs, &shared_lib_names);

  // find everything that is interesting
  std::set<std::string> dealii_shared_lib_names;
  for (std::set<std::string>::const_iterator p = shared_lib_names.begin();
       p != shared_lib_names.end(); ++p)
    if (p->find ("libdeal_II") != std::string::npos)
      dealii_shared_lib_names.insert (*p);

  // produce an error if we load deal.II more than once
  if (dealii_shared_lib_names.size() != 1)
    {
      std::ostringstream error;
      error << "........................................................\n"
            << "ASPECT currently links against different versions of the\n"
            << "deal.II library, namely the ones at these locations:\n";
      for (std::set<std::string>::const_iterator p = dealii_shared_lib_names.begin();
           p != dealii_shared_lib_names.end(); ++p)
        error << "  " << *p << '\n';
      error << "This can not work.\n\n";

      if (before_loading_shared_libs)
        error << "Since this is happening already before opening additional\n"
              << "shared libraries, this means that something must have gone\n"
              << "wrong when you configured deal.II and/or ASPECT. Please\n"
              << "contact the mailing lists for help.\n";
      else
        error << "Since this is happening after opening additional shared\n"
              << "library plugins, this likely means that you have compiled\n"
              << "ASPECT in release mode and the plugin in debug mode, or the\n"
              << "other way around. Please re-compile the plugin in the same\n"
              << "mode as ASPECT.\n";

      error << "........................................................\n";

      // if not success, then throw an exception: ExcMessage on processor 0,
      // QuietException on the others
      if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
        {
          AssertThrow (false, dealii::ExcMessage (error.str()));
        }
      else
        throw aspect::QuietException();
    }
#else
  // simply mark the argument as read, to avoid compiler warnings
  (void)before_loading_shared_libs;
#endif
}


#endif


// retrieve a list of shared libraries from the parameter file and
// dlopen them so that we can load plugins declared in them
void possibly_load_shared_libs (const std::string &parameters)
{
  using namespace dealii;


  const std::string shared_libs
    = get_last_value_of_parameter(parameters,
                                  "Additional shared libraries");
  if (shared_libs.size() > 0)
    {
#if ASPECT_USE_SHARED_LIBS==1
      // check up front whether the list of shared libraries is internally
      // consistent or whether we link, for whatever reason, with both the
      // debug and release versions of deal.II
      validate_shared_lib_list (true);

      const std::vector<std::string>
      shared_libs_list = Utilities::split_string_list (shared_libs);

      for (unsigned int i=0; i<shared_libs_list.size(); ++i)
        {
          if (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
            std::cout << "Loading shared library <"
                      << shared_libs_list[i]
                      << ">" << std::endl;

          void *handle = dlopen (shared_libs_list[i].c_str(), RTLD_LAZY);
          AssertThrow (handle != NULL,
                       ExcMessage (std::string("Could not successfully load shared library <")
                                   + shared_libs_list[i] + ">. The operating system reports "
                                   + "that the error is this: <"
                                   + dlerror() + ">."));

          // check again whether the list of shared libraries is
          // internally consistent or whether we link with both the
          // debug and release versions of deal.II. this may happen if
          // the plugin was compiled against the debug version of
          // deal.II but aspect itself against the release version, or
          // the other way around
          validate_shared_lib_list (false);

          // on systems where we can detect that both libdeal_II.so and
          // libdeal_II.g.so is loaded, the test above function above will
          // throw an exception and we will terminate. on the other hand, on
          // systems where we can't detect this we should at least mitigate
          // some of the ill effects -- in particular, make sure that
          // deallog is set to use the desired output depth since otherwise
          // we get lots of output from the linear solvers
          deallog.depth_console(0);
        }

      if (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
        std::cout << std::endl;
#else
      if (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
        {
          std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::cerr << "You can not load plugins through additional shared libraries " << std::endl
                    << "on systems where you link ASPECT as a static executable."
                    << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
        }
      std::exit (1);
#endif
    }
}

/*
 * Current implementation for reading from stdin requires use of a std::string,
 * so this function will read until the end of the stream
 */
std::string
read_until_end (std::istream &input)
{
  std::string result;
  while (input)
    {
      std::string line;
      std::getline(input, line);

      result += line + '\n';
    }
  return result;
}


/**
 * Let ParameterHandler parse the input file, here given as a string.
 * Since ParameterHandler unconditionally writes to the screen when it
 * finds something it doesn't like, we get massive amounts of output
 * in parallel computations since every processor writes the same
 * stuff to screen. To avoid this, let processor 0 parse the input
 * first and, if necessary, produce its output. Only if this
 * succeeds, also let the other processors read their input.
 *
 * In case of an error, we need to abort all processors without them
 * having read their data. This is done by throwing an exception of the
 * special class aspect::QuietException that we can catch in main() and terminate
 * the program quietly without generating other output.
 */
void
parse_parameters (const std::string &input_as_string,
                  dealii::ParameterHandler  &prm)
{
  // try reading on processor 0
  bool success = true;
  if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
    try
      {
        prm.parse_input_from_string(input_as_string.c_str());
      }
    catch (const dealii::ExceptionBase &e)
      {
        success = false;
        e.print_info(std::cerr);
        std::cerr << std::endl;
      }


  // broadcast the result. we'd like to do this with a bool
  // data type but MPI_C_BOOL is not part of old MPI standards.
  // so, do the broadcast in integers
  {
    int isuccess = (success ? 1 : 0);
    MPI_Bcast (&isuccess, 1, MPI_INT, 0, MPI_COMM_WORLD);
    success = (isuccess == 1);
  }

  // if not success, then throw an exception: ExcMessage on processor 0,
  // QuietException on the others
  if (success == false)
    {
      if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
        {
          AssertThrow(false, dealii::ExcMessage ("Invalid input parameter file."));
        }
      else
        throw aspect::QuietException();
    }

  // otherwise, processor 0 was ok reading the data, so we can expect the
  // other processors will be ok as well
  if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) != 0)
    {
      prm.parse_input_from_string(input_as_string.c_str());
    }
}



/**
 * Print help text
 */
void print_help()
{
  std::cout << "Usage: ./aspect [args] <parameter_file.prm>   (to read from an input file)"
            << std::endl
            << "    or ./aspect [args] --                     (to read parameters from stdin)"
            << std::endl
            << std::endl;
  std::cout << "    optional arguments [args]:"
            << std::endl
            << "       --version              (for information about library versions)"
            << std::endl
            << "       --help                 (for this usage help)"
            << std::endl
            << "       --output-xml           (print parameters in xml format to standard output and exit)"
            << std::endl
            << std::endl;
}


/**
 * Print information about the versions of underlying libraries.
 */
template <class Stream>
void print_version_information(Stream &stream)
{
  stream << "Version information of underlying libraries:\n"
         << "   . deal.II:    "
         << DEAL_II_PACKAGE_VERSION << '\n'
#ifndef ASPECT_USE_PETSC
         << "   . Trilinos:   "
         << DEAL_II_TRILINOS_VERSION_MAJOR    << '.'
         << DEAL_II_TRILINOS_VERSION_MINOR    << '.'
         << DEAL_II_TRILINOS_VERSION_SUBMINOR << '\n'
#else
         << "   . PETSc:      "
         << PETSC_VERSION_MAJOR    << '.'
         << PETSC_VERSION_MINOR    << '.'
         << PETSC_VERSION_SUBMINOR << '\n'
#endif
         << "   . p4est:      "
         << DEAL_II_P4EST_VERSION_MAJOR << '.'
         << DEAL_II_P4EST_VERSION_MINOR << '.'
         << DEAL_II_P4EST_VERSION_SUBMINOR << '\n'
         << std::endl;
}


int main (int argc, char *argv[])
{
  using namespace dealii;

  // disable the use of thread. if that is not what you want,
  // use numbers::invalid_unsigned_int instead of 1 to use as many threads
  // as deemed useful by TBB
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, /*n_threads =*/ 1);

#ifdef DEBUG
#ifdef ASPECT_USE_FP_EXCEPTIONS
  // enable floating point exceptions
  feenableexcept(FE_DIVBYZERO|FE_INVALID);
#endif
#endif

  try
    {
      deallog.depth_console(0);

      int current_idx = 1;
      const bool i_am_proc_0 = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      std::string prm_name = "";
      bool output_xml = false;

      // Loop over all command line arguments. Handle a number of special ones
      // starting with a dash, and then take the first non-special one as the
      // name of the input file. We will later check that there are no further
      // arguments left after that (though there may be with PETSc, see
      // below).
      while (current_idx<argc)
        {
          const std::string arg = argv[current_idx];
          ++current_idx;
          if (arg == "--output-xml")
            {
              output_xml = true;
            }
          else if (arg=="-h" || arg =="--help")
            {
              if (i_am_proc_0)
                {
                  print_aspect_header(std::cout);
                  print_help();
                }
              return 0;
            }
          else if (arg=="-v" || arg =="--version")
            {
              if (i_am_proc_0)
                {
                  print_aspect_header(std::cout);
                  print_version_information(std::cout);
                }
              return 0;
            }
          else
            {
              // Not a special argument, so we assume that this is the .prm
              // filename (or "--"). We can now break out of this loop because
              // we are not going to pass arguments passed after the filename
              prm_name = arg;
              break;
            }
        }


      // if no parameter given or somebody gave additional parameters,
      // show help and exit.
      // However, this does not work with PETSc because for PETSc, one
      // may pass any number of flags on the command line; unfortunately,
      // the PETSc initialization code (run through the call to
      // MPI_InitFinalize above) does not filter these out.
      if ((prm_name == "")
#ifndef ASPECT_USE_PETSC
          || (current_idx < argc)
#endif
         )
        {
          if (i_am_proc_0)
            {
              print_aspect_header(std::cout);
              print_help();
            }
          return 2;
        }

      // Print header
      if (i_am_proc_0 && !output_xml)
        {
          print_aspect_header(std::cout);
        }


      // See where to read input from, then do the reading and
      // put the contents of the input into a string.
      //
      // As stated above, treat "--" as special: as is common
      // on unix, treat it as a way to read input from stdin.
      std::string input_as_string;

      if (prm_name != "--")
        {
          std::ifstream parameter_file(prm_name.c_str());
          if (!parameter_file)
            {
              if (i_am_proc_0)
                AssertThrow(false, ExcMessage (std::string("Input parameter file <")
                                               + prm_name + "> not found."));
              return 3;
            }

          input_as_string = read_until_end (parameter_file);
        }
      else
        {
          // read parameters from stdin. unfortunately, if you do
          //    echo "abc" | mpirun -np 4 ./aspect
          // then only MPI process 0 gets the data. so we have to
          // read it there, then broadcast it to the other processors
          if (i_am_proc_0)
            {
              input_as_string = read_until_end (std::cin);
              int size = input_as_string.size()+1;
              MPI_Bcast (&size,
                         1,
                         MPI_INT,
                         /*root=*/0, MPI_COMM_WORLD);
              MPI_Bcast (const_cast<char *>(input_as_string.c_str()),
                         size,
                         MPI_CHAR,
                         /*root=*/0, MPI_COMM_WORLD);
            }
          else
            {
              // on this side, read what processor zero has broadcast about
              // the size of the input file. then create a buffer to put the
              // text in, get it from processor 0, and copy it to
              // input_as_string
              int size;
              MPI_Bcast (&size, 1,
                         MPI_INT,
                         /*root=*/0, MPI_COMM_WORLD);

              char *p = new char[size];
              MPI_Bcast (p, size,
                         MPI_CHAR,
                         /*root=*/0, MPI_COMM_WORLD);
              input_as_string = p;
              delete[] p;
            }
        }

      // Replace $ASPECT_SOURCE_DIR in the input so that include statements
      // like "include $ASPECT_SOURCE_DIR/tests/bla.prm" work.
      input_as_string = aspect::Utilities::expand_ASPECT_SOURCE_DIR(input_as_string);


      // try to determine the dimension we want to work in. the default
      // is 2, but if we find a line of the kind "set Dimension = ..."
      // then the last such line wins
      const unsigned int dim = get_dimension(input_as_string);

      // do the same with lines potentially indicating shared libs to
      // be loaded. these shared libs could contain additional module
      // instantiations for geometries, etc, that would then be
      // available as part of the possible parameters of the input
      // file, so they need to be loaded before we even start processing
      // the parameter file
      possibly_load_shared_libs (input_as_string);

      // now switch between the templates that code for 2d or 3d. it
      // would be nicer if we didn't have to duplicate code, but the
      // following needs to be known at compile time whereas the dimensionality
      // is only read at run-time
      ParameterHandler prm;

      switch (dim)
        {
          case 2:
          {
            aspect::Simulator<2>::declare_parameters(prm);
            parse_parameters (input_as_string, prm);

            if (output_xml)
              {
                if (i_am_proc_0)
                  prm.print_parameters(std::cout, ParameterHandler::XML);
              }
            else
              {
                aspect::Simulator<2> flow_problem(MPI_COMM_WORLD, prm);
                flow_problem.run();
              }
            break;
          }

          case 3:
          {
            aspect::Simulator<3>::declare_parameters(prm);
            parse_parameters (input_as_string, prm);

            if (output_xml)
              {
                if (i_am_proc_0)
                  prm.print_parameters(std::cout, ParameterHandler::XML);
              }
            else
              {
                aspect::Simulator<3> flow_problem(MPI_COMM_WORLD, prm);
                flow_problem.run();
              }

            break;
          }

          default:
            AssertThrow((dim >= 2) && (dim <= 3),
                        ExcMessage ("ASPECT can only be run in 2d and 3d but a "
                                    "different space dimension is given in the parameter file."));
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (aspect::QuietException &)
    {
      // Quietly treat an exception used on processors other than
      // root when we already know that processor 0 will generate
      // an exception. We do this to avoid creating too much
      // (duplicate) screen output.
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
