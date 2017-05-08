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
#include <aspect/adiabatic_conditions/interface.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/std_cxx11/tuple.h>

#include <list>


namespace aspect
{
  namespace AdiabaticConditions
  {
    template <int dim>
    Interface<dim>::~Interface ()
    {}

    template <int dim>
    void
    Interface<dim>::
    initialize ()
    {}

    template <int dim>
    void
    Interface<dim>::
    update ()
    {}

    template <int dim>
    void
    Interface<dim>::
    declare_parameters (dealii::ParameterHandler &)
    {}


    template <int dim>
    void
    Interface<dim>::parse_parameters (dealii::ParameterHandler &)
    {}


    template <int dim>
    void Interface<dim>::get_adiabatic_temperature_profile(std::vector<double> &values) const
    {
      const unsigned int num_slices = values.size();
      const double max_depth = this->get_geometry_model().maximal_depth();
      double depth = 0.0;

      for (unsigned int n = 0 ; n < num_slices; n++)
        {
          depth = n * max_depth / (num_slices-1);
          const Point<dim> p = this->get_geometry_model().representative_point(depth);
          values[n] = temperature(p);
        }
    }


    template <int dim>
    void Interface<dim>::get_adiabatic_pressure_profile(std::vector<double> &values) const
    {
      const unsigned int num_slices = values.size();
      const double max_depth = this->get_geometry_model().maximal_depth();
      double depth = 0.0;

      for (unsigned int n = 0 ; n < num_slices; n++)
        {
          depth = n * max_depth / (num_slices-1);
          const Point<dim> p = this->get_geometry_model().representative_point(depth);
          values[n] = pressure(p);
        }
    }

    template <int dim>
    void Interface<dim>::get_adiabatic_density_profile(std::vector<double> &values) const
    {
      const unsigned int num_slices = values.size();
      const double max_depth = this->get_geometry_model().maximal_depth();
      double depth = 0.0;

      for (unsigned int n = 0 ; n < num_slices; n++)
        {
          depth = n * max_depth / (num_slices-1);
          const Point<dim> p = this->get_geometry_model().representative_point(depth);
          values[n] = density(p);
        }
    }

    template <int dim>
    void Interface<dim>::get_adiabatic_density_derivative_profile(std::vector<double> &values) const
    {
      const unsigned int num_slices = values.size();
      const double max_depth = this->get_geometry_model().maximal_depth();
      double depth = 0.0;

      for (unsigned int n = 0 ; n < num_slices; n++)
        {
          depth = n * max_depth / (num_slices-1);
          const Point<dim> p = this->get_geometry_model().representative_point(depth);
          values[n] = density_derivative(p);
        }
    }


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
    register_adiabatic_conditions (const std::string &name,
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
    create_adiabatic_conditions (ParameterHandler &prm)
    {
      std::string model_name;
      prm.enter_subsection ("Adiabatic conditions model");
      {
        model_name = prm.get ("Model name");
      }
      prm.leave_subsection ();

      Interface<dim> *plugin = std_cxx11::get<dim>(registered_plugins).create_plugin (model_name,
                                                                                      "Adiabatic Conditions model::Model name");

      return plugin;

    }



    template <int dim>
    void
    declare_parameters (ParameterHandler &prm)
    {
      // declare the entry in the parameter file
      prm.enter_subsection ("Adiabatic conditions model");
      {
        const std::string pattern_of_names
          = std_cxx11::get<dim>(registered_plugins).get_pattern_of_names ();

        prm.declare_entry ("Model name", "initial profile",
                           Patterns::Selection (pattern_of_names),
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
      std::list<internal::Plugins::PluginList<AdiabaticConditions::Interface<2> >::PluginInfo> *
      internal::Plugins::PluginList<AdiabaticConditions::Interface<2> >::plugins = 0;
      template <>
      std::list<internal::Plugins::PluginList<AdiabaticConditions::Interface<3> >::PluginInfo> *
      internal::Plugins::PluginList<AdiabaticConditions::Interface<3> >::plugins = 0;
    }
  }

  namespace AdiabaticConditions
  {
#define INSTANTIATE(dim) \
  template class Interface<dim>; \
  \
  template \
  void \
  register_adiabatic_conditions<dim> (const std::string &, \
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
  create_adiabatic_conditions<dim> (ParameterHandler &prm);

    ASPECT_INSTANTIATE(INSTANTIATE)
  }
}
