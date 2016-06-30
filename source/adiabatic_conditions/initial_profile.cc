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


#include <aspect/global.h>
#include <aspect/adiabatic_conditions/initial_profile.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/box.h>

#include <deal.II/base/std_cxx11/bind.h>


namespace aspect
{
  namespace AdiabaticConditions
  {
    template <int dim>
    InitialProfile<dim>::InitialProfile()
      :
      initialized(false),
      n_points(1000),
      temperatures(n_points, -1),
      pressures(n_points, -1)
    {}

    template <int dim>
    void
    InitialProfile<dim>::initialize()
    {
      if (initialized)
        return;
      delta_z = this->get_geometry_model().maximal_depth() / (n_points-1);

      const unsigned int n_compositional_fields = this->n_compositional_fields();

      temperatures[0] = this->get_adiabatic_surface_temperature();
      pressures[0]    = this->get_surface_pressure();

      // now integrate downward using the explicit Euler method for simplicity
      //
      // note: p'(z) = rho(p,T) * |g|
      //       T'(z) = alpha |g| T / c_p
      double z;
      for (unsigned int i=1; i<n_points; ++i)
        {
          Assert (i < pressures.size(), ExcInternalError());
          Assert (i < temperatures.size(), ExcInternalError());

          z = double(i)/double(n_points-1)*this->get_geometry_model().maximal_depth();

          const Point<dim> representative_point = this->get_geometry_model().representative_point (z);
          const double delta_z = 1.0/double(n_points-1)*this->get_geometry_model().maximal_depth();
          Tensor <1,dim> g = this->get_gravity_model().gravity_vector(representative_point);

          MaterialModel::MaterialModelInputs<dim> in(1, n_compositional_fields);
          MaterialModel::MaterialModelOutputs<dim> out(1, n_compositional_fields);
          in.position[0] = representative_point;
          in.temperature[0] = temperatures[i-1];
          in.pressure[0] = pressures[i-1];
          in.velocity[0] = Tensor <1,dim> ();

          // we approximate the pressure gradient by extrapolating the values
          // from the two points above
          if (i>1)
            in.pressure_gradient[0] = g/(g.norm() != 0.0 ? g.norm() : 1.0)
                                      * (pressures[i-1] - pressures[i-2]) / delta_z;
          else
            in.pressure_gradient[0] = Tensor <1,dim> ();

          for (unsigned int c=0; c<n_compositional_fields; ++c)
            in.composition[0][c] = this->get_compositional_initial_conditions().initial_composition(representative_point, c);

          in.strain_rate.resize(0); // we do not need the viscosity
          this->get_material_model().evaluate(in, out);

          // get the magnitude of gravity. we assume
          // that gravity always points along the depth direction. this
          // may not strictly be true always but is likely a good enough
          // approximation here.
          const double density = out.densities[0];
          const double alpha = out.thermal_expansion_coefficients[0];
          // Handle the case that cp is zero (happens in simple Stokes test problems like sol_cx). By setting
          // 1/cp = 0.0 we will have a constant temperature profile with depth.
          const double one_over_cp = (out.specific_heat[0]>0.0) ? 1.0/out.specific_heat[0] : 0.0;
          const double gravity = this->get_gravity_model().gravity_vector(representative_point).norm();

          pressures[i] = pressures[i-1]
                         + density * gravity * delta_z;
          temperatures[i] = temperatures[i-1] * (1 +
                                                 alpha * gravity * delta_z * one_over_cp);
        }

      Assert (*std::min_element (pressures.begin(), pressures.end()) >=
              -std::numeric_limits<double>::epsilon() * pressures.size(),
              ExcMessage("Adiabatic InitialProfile encountered a negative pressure of "
                         + dealii::Utilities::to_string(*std::min_element (pressures.begin(), pressures.end()))));
      Assert (*std::min_element (temperatures.begin(), temperatures.end()) >=
              -std::numeric_limits<double>::epsilon() * temperatures.size(),
              ExcMessage("Adiabatic InitialProfile encountered a negative temperature."));

      initialized = true;
    }

    template <int dim>
    bool
    InitialProfile<dim>::is_initialized() const
    {
      return initialized;
    }

    template <int dim>
    void
    InitialProfile<dim>::update()
    {}


    template <int dim>
    double InitialProfile<dim>::pressure (const Point<dim> &p) const
    {
      const double z = this->get_geometry_model().depth(p);

      if (z >= this->get_geometry_model().maximal_depth())
        {
          Assert (z <= this->get_geometry_model().maximal_depth() + delta_z,
                  ExcInternalError());
          return pressures.back();
        }

      const unsigned int i = static_cast<unsigned int>(z/delta_z);
      Assert ((z/delta_z) >= 0, ExcInternalError());
      Assert (i+1 < pressures.size(), ExcInternalError());

      // now do the linear interpolation
      const double d=1.0+i-z/delta_z;
      Assert ((d>=0) && (d<=1), ExcInternalError());

      return d*pressures[i]+(1-d)*pressures[i+1];
    }



    template <int dim>
    double InitialProfile<dim>::temperature (const Point<dim> &p) const
    {
      const double z = this->get_geometry_model().depth(p);

      if (z >= this->get_geometry_model().maximal_depth())
        {
          Assert (z <= this->get_geometry_model().maximal_depth() + delta_z,
                  ExcInternalError());
          return temperatures.back();
        }

      const unsigned int i = static_cast<unsigned int>(z/delta_z);
      Assert ((z/delta_z) >= 0, ExcInternalError());
      Assert (i+1 < temperatures.size(), ExcInternalError());

      // now do the linear interpolation
      const double d=1.0+i-z/delta_z;
      Assert ((d>=0) && (d<=1), ExcInternalError());

      return d*temperatures[i]+(1-d)*temperatures[i+1];
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace AdiabaticConditions
  {
    ASPECT_REGISTER_ADIABATIC_CONDITIONS_MODEL(InitialProfile,
                                               "initial profile",
                                               "A model in which the adiabatic profile is "
                                               "calculated once at the start of the model run. "
                                               "The gravity is assumed to be in depth direction "
                                               "and the composition is evaluated at reference "
                                               "points, no lateral averaging is performed. "
                                               "All material parameters are used from the "
                                               "material model plugin.")
  }
}
