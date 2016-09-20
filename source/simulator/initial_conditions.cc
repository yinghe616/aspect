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
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/initial_conditions/interface.h>
#include <aspect/compositional_initial_conditions/interface.h>
#include <aspect/postprocess/tracers.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>


namespace aspect
{

  template <int dim>
  void Simulator<dim>::set_initial_temperature_and_compositional_fields ()
  {
    // create a fully distributed vector since we
    // need to write into it and we can not
    // write into vectors with ghost elements
    LinearAlgebra::BlockVector initial_solution;
    double max_sum_comp = 0.0;

    // we need to track whether we need to normalize the totality of fields
    bool normalize_composition = false;

    initial_solution.reinit(system_rhs, false);

    // below, we would want to call VectorTools::interpolate on the
    // entire FESystem. there currently is no way to restrict the
    // interpolation operations to only a subset of vector
    // components (oversight in deal.II?), specifically to the
    // temperature component. this causes more work than necessary
    // but worse yet, it doesn't work for the DGP(q) pressure element
    // if we use a locally conservative formulation since there the
    // pressure element is non-interpolating (we get an exception
    // even though we are, strictly speaking, not interested in
    // interpolating the pressure; but, as mentioned, there is no way
    // to tell VectorTools::interpolate that)
    //
    // to work around this problem, the following code is essentially
    // a (simplified) copy of the code in VectorTools::interpolate
    // that only works on the temperature component
    //
    //TODO: it would be great if we had a cleaner way than iterating to 1+n_fields.
    // Additionally, the n==1 logic for normalization at the bottom is not pretty.
    for (unsigned int n=0; n<1+parameters.n_compositional_fields; ++n)
      {
        AdvectionField advf = ((n == 0) ? AdvectionField::temperature()
                               : AdvectionField::composition(n-1));

        const unsigned int base_element = advf.base_element(introspection);

        // get the temperature/composition support points
        const std::vector<Point<dim> > support_points
          = finite_element.base_element(base_element).get_unit_support_points();
        Assert (support_points.size() != 0,
                ExcInternalError());

        // create an FEValues object with just the temperature/composition element
        FEValues<dim> fe_values (*mapping, finite_element,
                                 support_points,
                                 update_quadrature_points);

        std::vector<types::global_dof_index> local_dof_indices (finite_element.dofs_per_cell);

#if 0 //DEAL_II_VERSION_GTE(8,5,0)
        const VectorFunctionFromScalarFunctionObject<dim, double> &advf_init_function =
          (advf.is_temperature()
           ?
           VectorFunctionFromScalarFunctionObject<dim, double>(std_cxx11::bind(&InitialConditions::Interface<dim>::initial_temperature,
                                                                               std::ref(*initial_conditions),
                                                                               std_cxx11::_1),
                                                               introspection.component_indices.temperature,
                                                               introspection.n_components)
           :
           VectorFunctionFromScalarFunctionObject<dim, double>(std_cxx11::bind(&CompositionalInitialConditions::Interface<dim>::initial_composition,
                                                                               std::ref(*compositional_initial_conditions),
                                                                               std_cxx11::_1,
                                                                               n-1),
                                                               introspection.component_indices.compositional_fields[n-1],
                                                               introspection.n_components));

        const ComponentMask advf_mask =
          (advf.is_temperature()
           ?
           introspection.component_masks.temperature
           :
           introspection.component_masks.compositional_fields[n-1]);

        VectorTools::interpolate(*mapping,
                                 dof_handler,
                                 advf_init_function,
                                 initial_solution,
                                 advf_mask);

        if (parameters.normalized_fields.size()>0 && n==1)
          for (typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
               cell != dof_handler.end(); ++cell)
            if (cell->is_locally_owned())
              {
                fe_values.reinit (cell);

                // Go through the support points for each dof
                for (unsigned int i=0; i<finite_element.base_element(base_element).dofs_per_cell; ++i)
                  {
                    // if it is specified in the parameter file that the sum of all compositional fields
                    // must not exceed one, this should be checked
                    double sum = 0;
                    for (unsigned int m=0; m<parameters.normalized_fields.size(); ++m)
                      sum += compositional_initial_conditions->initial_composition(fe_values.quadrature_point(i),
                                                                                   parameters.normalized_fields[m]);
                    if (std::abs(sum) > 1.0+std::numeric_limits<double>::epsilon())
                      {
                        max_sum_comp = std::max(sum, max_sum_comp);
                        normalize_composition = true;
                      }
                  }
              }
#else
        for (typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
             cell != dof_handler.end(); ++cell)
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);

              // go through the temperature/composition dofs and set their global values
              // to the temperature/composition field interpolated at these points
              cell->get_dof_indices (local_dof_indices);
              for (unsigned int i=0; i<finite_element.base_element(base_element).dofs_per_cell; ++i)
                {
                  const unsigned int system_local_dof
                    = finite_element.component_to_system_index(advf.component_index(introspection),
                                                               /*dof index within component=*/i);

                  const double value =
                    (advf.is_temperature()
                     ?
                     initial_conditions->initial_temperature(fe_values.quadrature_point(i))
                     :
                     compositional_initial_conditions->initial_composition(fe_values.quadrature_point(4),n-1));
                  initial_solution(local_dof_indices[system_local_dof]) = value;

                  // if it is specified in the parameter file that the sum of all compositional fields
                  // must not exceed one, this should be checked
                  if (parameters.normalized_fields.size()>0 && n == 1)
                    {
                      double sum = 0;
                      for (unsigned int m=0; m<parameters.normalized_fields.size(); ++m)
                        sum += compositional_initial_conditions->initial_composition(fe_values.quadrature_point(i),
                                                                                     parameters.normalized_fields[m]);
                      if (std::abs(sum) > 1.0+std::numeric_limits<double>::epsilon())
                        {
                          max_sum_comp = std::max(sum, max_sum_comp);
                          normalize_composition = true;
                        }
                    }

                }
            }
#endif

        initial_solution.compress(VectorOperation::insert);

        // if at least one processor decides that it needs
        // to normalize, do the same on all processors.
        if (Utilities::MPI::max (normalize_composition ? 1 : 0,
                                 mpi_communicator)
            == 1)
          {
            const double global_max
              = Utilities::MPI::max (max_sum_comp, mpi_communicator);

            if (n==1)
              pcout << "Sum of compositional fields is not one, fields will be normalized"
                    << std::endl;

            for (unsigned int m=0; m<parameters.normalized_fields.size(); ++m)
              if (n-1==parameters.normalized_fields[m])
                initial_solution.block(introspection.block_indices.compositional_fields[n-1]) /= global_max;
          }
      }

    // then apply constraints and copy the
    // result into vectors with ghost elements. to do so,
    // we need the current constraints to be correct for
    // the current time
    compute_current_constraints ();
    current_constraints.distribute(initial_solution);

    // Now copy the temperature and initial composition blocks into the solution variables

    for (unsigned int n=0; n<1+parameters.n_compositional_fields; ++n)
      {
        AdvectionField advf = ((n == 0) ? AdvectionField::temperature()
                               : AdvectionField::composition(n-1));

        const unsigned int blockidx = advf.block_index(introspection);

        solution.block(blockidx) = initial_solution.block(blockidx);
        old_solution.block(blockidx) = initial_solution.block(blockidx);
        old_old_solution.block(blockidx) = initial_solution.block(blockidx);
      }
  }


  template <int dim>
  void Simulator<dim>::interpolate_particle_properties (const AdvectionField &advection_field)
  {
    // below, we would want to call VectorTools::interpolate on the
    // entire FESystem. there currently is no way to restrict the
    // interpolation operations to only a subset of vector
    // components (oversight in deal.II?), specifically to the
    // temperature component. this causes more work than necessary
    // but worse yet, it doesn't work for the DGP(q) pressure element
    // if we use a locally conservative formulation since there the
    // pressure element is non-interpolating (we get an exception
    // even though we are, strictly speaking, not interested in
    // interpolating the pressure; but, as mentioned, there is no way
    // to tell VectorTools::interpolate that)
    //
    // to work around this problem, the following code is essentially
    // a (simplified) copy of the code in VectorTools::interpolate
    // that only works on the given component

    // create a fully distributed vector since we
    // need to write into it and we can not
    // write into vectors with ghost elements

    const Postprocess::Tracers<dim> *tracer_postprocessor = postprocess_manager.template find_postprocessor<Postprocess::Tracers<dim> >();

    AssertThrow(tracer_postprocessor != 0,
                ExcMessage("Did not find the <tracers> postprocessor when trying to interpolate particle properties."));

    const std::multimap<aspect::Particle::types::LevelInd, Particle::Particle<dim> > *particles = &tracer_postprocessor->get_particle_world().get_particles();
    const Particle::Interpolator::Interface<dim> *particle_interpolator = &tracer_postprocessor->get_particle_world().get_interpolator();
    const Particle::Property::Manager<dim> *particle_property_manager = &tracer_postprocessor->get_particle_world().get_property_manager();

    unsigned int particle_property;

    if (parameters.mapped_particle_properties.size() != 0)
      {
        const std::pair<std::string,unsigned int> particle_property_and_component = parameters.mapped_particle_properties.find(advection_field.compositional_variable)->second;

        particle_property = particle_property_manager->get_data_info().get_position_by_field_name(particle_property_and_component.first)
                            + particle_property_and_component.second;
      }
    else
      {
        particle_property = std::count(introspection.compositional_field_methods.begin(),
                                       introspection.compositional_field_methods.begin() + advection_field.compositional_variable,
                                       Parameters<dim>::AdvectionFieldMethod::particles);
        AssertThrow(particle_property <= particle_property_manager->get_data_info().n_components(),
                    ExcMessage("Can not automatically match particle properties to fields, because there are"
                               "more fields that are marked as particle advected than particle properties"));
      }

    LinearAlgebra::BlockVector tracer_solution;

    tracer_solution.reinit(system_rhs, false);

    const unsigned int base_element = advection_field.base_element(introspection);

    // get the temperature/composition support points
    const std::vector<Point<dim> > support_points
      = finite_element.base_element(base_element).get_unit_support_points();
    Assert (support_points.size() != 0,
            ExcInternalError());

    // create an FEValues object with just the temperature/composition element
    FEValues<dim> fe_values (*mapping, finite_element,
                             support_points,
                             update_quadrature_points);

    std::vector<types::global_dof_index> local_dof_indices (finite_element.dofs_per_cell);

    for (typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          const std::vector<Point<dim> > quadrature_points = fe_values.get_quadrature_points();

          const std::vector<std::vector<double> > tracer_properties =
            particle_interpolator->properties_at_points(*particles,quadrature_points,cell);

          // go through the temperature/composition dofs and set their global values
          // to the particle field interpolated at these points
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<finite_element.base_element(base_element).dofs_per_cell; ++i)
            {
              const unsigned int system_local_dof
                = finite_element.component_to_system_index(advection_field.component_index(introspection),
                                                           /*dof index within component=*/i);


              const double value = tracer_properties[i][particle_property];
              tracer_solution(local_dof_indices[system_local_dof]) = value;
            }
        }

    tracer_solution.compress(VectorOperation::insert);

    // we should not have written at all into any of the blocks with
    // the exception of the current temperature or composition block
    for (unsigned int b=0; b<tracer_solution.n_blocks(); ++b)
      if (b != advection_field.block_index(introspection))
        Assert (tracer_solution.block(b).l2_norm() == 0,
                ExcInternalError());

    // copy temperature/composition block only
    const unsigned int blockidx = advection_field.block_index(introspection);
    solution.block(blockidx) = tracer_solution.block(blockidx);
    old_solution.block(blockidx) = tracer_solution.block(blockidx);
    old_old_solution.block(blockidx) = tracer_solution.block(blockidx);
  }


  template <int dim>
  void Simulator<dim>::compute_initial_pressure_field ()
  {
    // Note that this code will overwrite the velocity solution with 0 if
    // velocity and pressure are in the same block (i.e., direct solver is
    // used). As the velocity is all zero anyway, this is currently not a
    // problem.

    // we'd like to interpolate the initial pressure onto the pressure
    // variable but that's a bit involved because the pressure may either
    // be an FE_Q (for which we can interpolate) or an FE_DGP (for which
    // we can't since the element has no nodal basis.
    //
    // fortunately, in the latter case, the element is discontinuous and
    // we can compute a local projection onto the pressure space
    if (parameters.use_locally_conservative_discretization == false)
      {
        // allocate a vector that is distributed but doesn't have
        // ghost elements (vectors with ghost elements are not
        // writable); the stokes_rhs vector is a valid template for
        // this kind of thing. interpolate into it and later copy it into the
        // solution vector that does have the necessary ghost elements
        LinearAlgebra::BlockVector system_tmp;
        system_tmp.reinit (system_rhs);

        // interpolate the pressure given by the adiabatic conditions
        // object onto the solution space. note that interpolate
        // wants a function that represents all components of the
        // solution vector, so create such a function object
        // that is simply zero for all velocity components
        const unsigned int pressure_comp =
          parameters.include_melt_transport ?
          introspection.variable("fluid pressure").first_component_index
          :
          introspection.component_indices.pressure;

        VectorTools::interpolate (*mapping, dof_handler,
                                  VectorFunctionFromScalarFunctionObject<dim> (std_cxx11::bind (&AdiabaticConditions::Interface<dim>::pressure,
                                                                               std_cxx11::cref (*adiabatic_conditions),
                                                                               std_cxx11::_1),
                                                                               pressure_comp,
                                                                               introspection.n_components),
                                  system_tmp);

        // we may have hanging nodes, so apply constraints
        constraints.distribute (system_tmp);

        const unsigned int pressure_block = (parameters.include_melt_transport ?
                                             introspection.variable("fluid pressure").block_index
                                             : introspection.block_indices.pressure);
        old_solution.block(pressure_block) = system_tmp.block(pressure_block);
      }
    else
      {
        // implement the local projection for the discontinuous pressure
        // element. this is only going to work if, indeed, the element
        // is discontinuous
        Assert (finite_element.base_element(introspection.base_elements.pressure).dofs_per_face == 0,
                ExcNotImplemented());

        LinearAlgebra::BlockVector system_tmp;
        system_tmp.reinit (system_rhs);

        QGauss<dim> quadrature(parameters.stokes_velocity_degree+1);
        UpdateFlags update_flags = UpdateFlags(update_values   |
                                               update_quadrature_points |
                                               update_JxW_values);

        FEValues<dim> fe_values (*mapping, finite_element, quadrature, update_flags);

        const unsigned int
        dofs_per_cell = fe_values.dofs_per_cell,
        n_q_points    = fe_values.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        Vector<double> cell_vector (dofs_per_cell);
        Vector<double> local_projection (dofs_per_cell);
        FullMatrix<double> local_mass_matrix (dofs_per_cell, dofs_per_cell);

        std::vector<double> rhs_values(n_q_points);

        ScalarFunctionFromFunctionObject<dim>
        adiabatic_pressure (std_cxx11::bind (&AdiabaticConditions::Interface<dim>::pressure,
                                             std_cxx11::cref(*adiabatic_conditions),
                                             std_cxx11::_1));


        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices (local_dof_indices);
              fe_values.reinit(cell);

              adiabatic_pressure.value_list (fe_values.get_quadrature_points(),
                                             rhs_values);

              cell_vector = 0;
              local_mass_matrix = 0;
              for (unsigned int point=0; point<n_q_points; ++point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  {
                    if (finite_element.system_to_component_index(i).first == dim)
                      cell_vector(i)
                      +=
                        rhs_values[point] *
                        fe_values[introspection.extractors.pressure].value(i,point) *
                        fe_values.JxW(point);

                    // populate the local matrix; create the pressure mass matrix
                    // in the pressure pressure block and the identity matrix
                    // for all other variables so that the whole thing remains
                    // invertible
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                      if ((finite_element.system_to_component_index(i).first == introspection.component_indices.pressure)
                          &&
                          (finite_element.system_to_component_index(j).first == introspection.component_indices.pressure))
                        local_mass_matrix(j,i) += (fe_values[introspection.extractors.pressure].value(i,point) *
                                                   fe_values[introspection.extractors.pressure].value(j,point) *
                                                   fe_values.JxW(point));
                      else if (i == j)
                        local_mass_matrix(i,j) = 1;
                  }

              // now invert the local mass matrix and multiply it with the rhs
              local_mass_matrix.gauss_jordan();
              local_mass_matrix.vmult (local_projection, cell_vector);

              // then set the global solution vector to the values just computed
              cell->set_dof_values (local_projection, system_tmp);
            }

        old_solution.block(introspection.block_indices.pressure) = system_tmp.block(introspection.block_indices.pressure);
      }

    // normalize the pressure in such a way that the surface pressure
    // equals a known and desired value
    normalize_pressure(old_solution);

    // set all solution vectors to the same value as the previous solution
    solution = old_solution;
    old_old_solution = old_solution;
  }
}



// explicit instantiation of the functions we implement in this file
namespace aspect
{
#define INSTANTIATE(dim) \
  template void Simulator<dim>::set_initial_temperature_and_compositional_fields(); \
  template void Simulator<dim>::compute_initial_pressure_field(); \
  template void Simulator<dim>::interpolate_particle_properties(const AdvectionField &);


  ASPECT_INSTANTIATE(INSTANTIATE)
}
