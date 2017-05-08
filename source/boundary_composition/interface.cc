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


#include <aspect/global.h>
#include <aspect/boundary_composition/interface.h>

#include <aspect/utilities.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/base/std_cxx11/tuple.h>

#include <list>


namespace aspect
{
  namespace BoundaryComposition
  {
    template <int dim>
    Interface<dim>::~Interface ()
    {}

    template <int dim>
    void
    Interface<dim>::update ()
    {}

    template <int dim>
    void
    Interface<dim>::initialize ()
    {}

    template <int dim>
    double
    Interface<dim>::boundary_composition (const types::boundary_id /*boundary_indicator*/,
                                          const Point<dim>        &/*position*/,
                                          const unsigned int       /*compositional_field*/) const
    {
      AssertThrow(false,
                  ExcMessage("The boundary composition plugin has to implement a function called 'composition' "
                             "with four arguments or a function 'boundary_composition' with three arguments. "
                             "The function with four arguments is deprecated and will "
                             "be removed in a later version of ASPECT."));
      return numbers::signaling_nan<double>();
    }

    template <int dim>
    void
    Interface<dim>::
    declare_parameters (dealii::ParameterHandler &)
    {}


    template <int dim>
    void
    Interface<dim>::parse_parameters (dealii::ParameterHandler &)
    {}


// -------------------------------- Deal with registering models and automating
// -------------------------------- their setup and selection at run time

    namespace
    {
      std_cxx11::tuple
      <void *,
      void *,
      aspect::internal::Plugins::PluginList<Interface<2> >,
      aspect::internal::Plugins::PluginList<Interface<3> > > registered_plugins;
    }



    template <int dim>
    void
    register_boundary_composition (const std::string &name,
                                   const std::string &description,
                                   void (*declare_parameters_function) (ParameterHandler &),
                                   Interface<dim> *(*factory_function) ())
    {
      std_cxx11::get<dim>(registered_plugins).register_plugin (name,
                                                               description,
                                                               declare_parameters_function,
                                                               factory_function);
    }


    template <int dim>
    Interface<dim> *
    create_boundary_composition (ParameterHandler &prm)
    {
      std::string model_name;
      prm.enter_subsection ("Boundary composition model");
      {
        model_name = prm.get ("Model name");
      }
      prm.leave_subsection ();

      // If one sets the model name to an empty string in the input file,
      // ParameterHandler produces an error while reading the file. However,
      // if one omits specifying any model name at all (not even setting it to
      // the empty string) then the value we get here is the empty string. If
      // we don't catch this case here, we end up with awkward downstream
      // errors because the value obviously does not conform to the Pattern.
      AssertThrow(model_name != "unspecified",
                  ExcMessage("You need to select a boundary model for the composition "
                             "('set Model name' in 'subsection Boundary composition model')."));

      return std_cxx11::get<dim>(registered_plugins).create_plugin (model_name,
                                                                    "Boundary composition model::Model name");
    }



    template <int dim>
    void
    declare_parameters (ParameterHandler &prm)
    {
      // declare the entry in the parameter file
      prm.enter_subsection ("Boundary composition model");
      {
        const std::string pattern_of_names
          = std_cxx11::get<dim>(registered_plugins).get_pattern_of_names ();
        prm.declare_entry ("Model name", "unspecified",
                           Patterns::Selection (pattern_of_names+"|unspecified"),
                           "Select one of the following models:\n\n"
                           +
                           std_cxx11::get<dim>(registered_plugins).get_description_string());
      }
      prm.leave_subsection ();

      std_cxx11::get<dim>(registered_plugins).declare_parameters (prm);
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace internal
  {
    namespace Plugins
    {
      template <>
      std::list<internal::Plugins::PluginList<BoundaryComposition::Interface<2> >::PluginInfo> *
      internal::Plugins::PluginList<BoundaryComposition::Interface<2> >::plugins = 0;
      template <>
      std::list<internal::Plugins::PluginList<BoundaryComposition::Interface<3> >::PluginInfo> *
      internal::Plugins::PluginList<BoundaryComposition::Interface<3> >::plugins = 0;
    }
  }

  namespace BoundaryComposition
  {
#define INSTANTIATE(dim) \
  template class Interface<dim>; \
  \
  template \
  void \
  register_boundary_composition<dim> (const std::string &, \
                                      const std::string &, \
                                      void ( *) (ParameterHandler &), \
                                      Interface<dim> *( *) ()); \
  \
  template  \
  void \
  declare_parameters<dim> (ParameterHandler &); \
  \
  template \
  Interface<dim> * \
  create_boundary_composition<dim> (ParameterHandler &prm);

    ASPECT_INSTANTIATE(INSTANTIATE)
  }
}
