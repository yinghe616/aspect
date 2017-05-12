/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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
#include <aspect/global.h>
#include <aspect/utilities.h>
#include <aspect/melt.h>
#include <aspect/free_surface.h>

#include <deal.II/base/parameter_handler.h>

#include <sys/stat.h>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>

namespace aspect
{
  template <int dim>
  Parameters<dim>::Parameters (ParameterHandler &prm,
                               MPI_Comm mpi_communicator)
  {
    parse_parameters (prm, mpi_communicator);
  }


  template <int dim>
  void
  Parameters<dim>::
  declare_parameters (ParameterHandler &prm)
  {
    prm.declare_entry ("Dimension", "2",
                       Patterns::Integer (2,4),
                       "The number of space dimensions you want to run this program in. "
                       "ASPECT can run in 2 and 3 space dimensions.");
    prm.declare_entry ("Additional shared libraries", "",
                       Patterns::List (Patterns::FileName()),
                       "A list of names of additional shared libraries that should be loaded "
                       "upon starting up the program. The names of these files can contain absolute "
                       "or relative paths (relative to the directory in which you call ASPECT). "
                       "In fact, file names that are do not contain any directory "
                       "information (i.e., only the name of a file such as <myplugin.so> "
                       "will not be found if they are not located in one of the directories "
                       "listed in the \\texttt{LD_LIBRARY_PATH} environment variable. In order "
                       "to load a library in the current directory, use <./myplugin.so> "
                       "instead."
                       "\n\n"
                       "The typical use of this parameter is so that you can implement "
                       "additional plugins in your own directories, rather than in the ASPECT "
                       "source directories. You can then simply compile these plugins into a "
                       "shared library without having to re-compile all of ASPECT. See the "
                       "section of the manual discussing writing extensions for more "
                       "information on how to compile additional files into a shared "
                       "library.");

    prm.declare_entry ("Resume computation", "false",
                       Patterns::Selection ("true|false|auto"),
                       "A flag indicating whether the computation should be resumed from "
                       "a previously saved state (if true) or start from scratch (if false). "
                       "If auto is selected, models will be resumed if there is an existing "
                       "checkpoint file, otherwise started from scratch.");

    prm.declare_entry ("Max nonlinear iterations", "10",
                       Patterns::Integer (0),
                       "The maximal number of nonlinear iterations to be performed.");

    prm.declare_entry ("Max nonlinear iterations in pre-refinement", boost::lexical_cast<std::string>(std::numeric_limits<int>::max()),
                       Patterns::Integer (0),
                       "The maximal number of nonlinear iterations to be performed in the pre-refinement "
                       "steps. This does not include the last refinement step before moving to timestep 1. "
                       "When this parameter has a larger value than max nonlinear iterations, the latter is used.");

    prm.declare_entry ("Start time", "0",
                       Patterns::Double (),
                       "The start time of the simulation. Units: Years if the "
                       "'Use years in output instead of seconds' parameter is set; "
                       "seconds otherwise.");

    prm.declare_entry ("Timing output frequency", "100",
                       Patterns::Integer(0),
                       "How frequently in timesteps to output timing information. This is "
                       "generally adjusted only for debugging and timing purposes. If the "
                       "value is set to zero it will also output timing information at the "
                       "initiation timesteps.");

    prm.declare_entry ("Use years in output instead of seconds", "true",
                       Patterns::Bool (),
                       "When computing results for mantle convection simulations, "
                       "it is often difficult to judge the order of magnitude of results "
                       "when they are stated in MKS units involving seconds. Rather, "
                       "some kinds of results such as velocities are often stated in "
                       "terms of meters per year (or, sometimes, centimeters per year). "
                       "On the other hand, for non-dimensional computations, one wants "
                       "results in their natural unit system as used inside the code. "
                       "If this flag is set to 'true' conversion to years happens; if "
                       "it is 'false', no such conversion happens. Note that when 'true', "
                       "some input such as prescribed velocities should also use years "
                       "instead of seconds.");

    prm.declare_entry ("CFL number", "1.0",
                       Patterns::Double (0),
                       "In computations, the time step $k$ is chosen according to "
                       "$k = c \\min_K \\frac {h_K} {\\|u\\|_{\\infty,K} p_T}$ where $h_K$ is the "
                       "diameter of cell $K$, and the denominator is the maximal magnitude "
                       "of the velocity on cell $K$ times the polynomial degree $p_T$ of the "
                       "temperature discretization. The dimensionless constant $c$ is called the "
                       "CFL number in this program. For time discretizations that have explicit "
                       "components, $c$ must be less than a constant that depends on the "
                       "details of the time discretization and that is no larger than one. "
                       "On the other hand, for implicit discretizations such as the one chosen "
                       "here, one can choose the time step as large as one wants (in particular, "
                       "one can choose $c>1$) though a CFL number significantly larger than "
                       "one will yield rather diffusive solutions. Units: None.");

    prm.declare_entry ("Maximum time step",
                       /* boost::lexical_cast<std::string>(std::numeric_limits<double>::max() /
                                                           year_in_seconds) = */ "5.69e+300",
                       Patterns::Double (0),
                       "Set a maximum time step size for the solver to use. Generally the time step "
                       "based on the CFL number should be sufficient, but for complicated models "
                       "or benchmarking it may be useful to limit the time step to some value. "
                       "The default value is a value so that when converted from years into seconds "
                       "it equals the largest number representable by a floating "
                       "point number, implying an unlimited time step."
                       "Units: Years or seconds, depending on the ``Use years "
                       "in output instead of seconds'' parameter.");

    prm.declare_entry ("Use conduction timestep", "false",
                       Patterns::Bool (),
                       "Mantle convection simulations are often focused on convection "
                       "dominated systems. However, these codes can also be used to "
                       "investigate systems where heat conduction plays a dominant role. "
                       "This parameter indicates whether the simulator should also use "
                       "heat conduction in determining the length of each time step.");

    prm.declare_entry ("Nonlinear solver scheme", "IMPES",
                       Patterns::Selection ("IMPES|iterated IMPES|iterated Stokes|Stokes only|Advection only"),
                       "The kind of scheme used to resolve the nonlinearity in the system. "
                       "'IMPES' is the classical IMplicit Pressure Explicit Saturation scheme "
                       "in which ones solves the temperatures and Stokes equations exactly "
                       "once per time step, one after the other. The 'iterated IMPES' scheme "
                       "iterates this decoupled approach by alternating the solution of the "
                       "temperature and Stokes systems. The 'iterated Stokes' scheme solves "
                       "the temperature equation once at the beginning of each time step "
                       "and then iterates out the solution of the Stokes equation. The 'Stokes only' "
                       "scheme only solves the Stokes system and ignores compositions and the "
                       "temperature equation (careful, the material model must not depend on "
                       "the temperature; mostly useful for Stokes benchmarks). The 'Advection only'"
                       "scheme only solves the temperature and other advection systems and instead "
                       "of solving for the Stokes system, a prescribed velocity and pressure is "
                       "used");

    prm.declare_entry ("Nonlinear solver tolerance", "1e-5",
                       Patterns::Double(0,1),
                       "A relative tolerance up to which the nonlinear solver "
                       "will iterate. This parameter is only relevant if "
                       "Nonlinear solver scheme is set to 'iterated Stokes' or "
                       "'iterated IMPES'.");

    prm.declare_entry ("Pressure normalization", "surface",
                       Patterns::Selection ("surface|volume|no"),
                       "If and how to normalize the pressure after the solution step. "
                       "This is necessary because depending on boundary conditions, "
                       "in many cases the pressure is only determined by the model "
                       "up to a constant. On the other hand, we often would like to "
                       "have a well-determined pressure, for example for "
                       "table lookups of material properties in models "
                       "or for comparing solutions. If the given value is `surface', then "
                       "normalization at the end of each time steps adds a constant value "
                       "to the pressure in such a way that the average pressure at the surface "
                       "of the domain is what is set in the `Surface pressure' parameter; "
                       "the surface of the domain is determined by asking "
                       "the geometry model whether a particular face of the geometry has a zero "
                       "or small `depth'. If the value of this parameter is `volume' then the "
                       "pressure is normalized so that the domain average is zero. If `no' is "
                       "given, the no pressure normalization is performed.");

    prm.declare_entry ("Surface pressure", "0",
                       Patterns::Double(),
                       "The value the pressure is normalized to in each time step when "
                       "`Pressure normalization' is set to `surface' with default value 0. "
                       "This setting is ignored in all other cases."
                       "\n\n"
                       "The mathematical equations that describe thermal convection "
                       "only determine the pressure up to an arbitrary constant. On "
                       "the other hand, for comparison and for looking up material "
                       "parameters it is important that the pressure be normalized "
                       "somehow. We do this by enforcing a particular average pressure "
                       "value at the surface of the domain, where the geometry model "
                       "determines where the surface is. This parameter describes what "
                       "this average surface pressure value is supposed to be. By "
                       "default, it is set to zero, but one may want to choose a "
                       "different value for example for simulating only the volume "
                       "of the mantle below the lithosphere, in which case the surface "
                       "pressure should be the lithostatic pressure at the bottom "
                       "of the lithosphere."
                       "\n\n"
                       "For more information, see the section in the manual that discusses "
                       "the general mathematical model.");

    prm.declare_entry ("Adiabatic surface temperature", "0",
                       Patterns::Double(),
                       "In order to make the problem in the first time step easier to "
                       "solve, we need a reasonable guess for the temperature and pressure. "
                       "To obtain it, we use an adiabatic pressure and temperature field. "
                       "This parameter describes what the `adiabatic' temperature would "
                       "be at the surface of the domain (i.e. at depth zero). Note "
                       "that this value need not coincide with the boundary condition "
                       "posed at this point. Rather, the boundary condition may differ "
                       "significantly from the adiabatic value, and then typically "
                       "induce a thermal boundary layer."
                       "\n\n"
                       "For more information, see the section in the manual that discusses "
                       "the general mathematical model.");

    prm.declare_entry ("Output directory", "output",
                       Patterns::DirectoryName(),
                       "The name of the directory into which all output files should be "
                       "placed. This may be an absolute or a relative path.");

    prm.declare_entry ("Use direct solver for Stokes system", "false",
                       Patterns::Bool(),
                       "If set to true the linear system for the Stokes equation will "
                       "be solved using Trilinos klu, otherwise an iterative Schur "
                       "complement solver is used. The direct solver is only efficient "
                       "for small problems.");

    prm.declare_entry ("Linear solver tolerance", "1e-7",
                       Patterns::Double(0,1),
                       "A relative tolerance up to which the linear Stokes systems in each "
                       "time or nonlinear step should be solved. The absolute tolerance will "
                       "then be $\\| M x_0 - F \\| \\cdot \\text{tol}$, where $x_0 = (0,p_0)$ "
                       "is the initial guess of the pressure, $M$ is the system matrix, "
                       "F is the right-hand side, and tol is the parameter specified here. "
                       "We include the initial guess of the pressure "
                       "to remove the dependency of the tolerance on the static pressure. "
                       "A given tolerance value of 1 would "
                       "mean that a zero solution vector is an acceptable solution "
                       "since in that case the norm of the residual of the linear "
                       "system equals the norm of the right hand side. A given "
                       "tolerance of 0 would mean that the linear system has to be "
                       "solved exactly, since this is the only way to obtain "
                       "a zero residual."
                       "\n\n"
                       "In practice, you should choose the value of this parameter "
                       "to be so that if you make it smaller the results of your "
                       "simulation do not change any more (qualitatively) whereas "
                       "if you make it larger, they do. For most cases, the default "
                       "value should be sufficient. In fact, a tolerance of 1e-4 "
                       "might be accurate enough.");

    prm.declare_entry ("Linear solver A block tolerance", "1e-2",
                       Patterns::Double(0,1),
                       "A relative tolerance up to which the approximate inverse of the $A$ block "
                       "of the Stokes system is computed. This approximate $A$ is used in the "
                       "preconditioning used in the GMRES solver. The exact definition of this "
                       "block preconditioner for the Stokes equation can be found in "
                       "\\cite{KHB12}.");

    prm.declare_entry ("Linear solver S block tolerance", "1e-6",
                       Patterns::Double(0,1),
                       "A relative tolerance up to which the approximate inverse of the $S$ block "
                       "(i.e., the Schur complement matrix $S = BA^{-1}B^{T}$) of the Stokes "
                       "system is computed. This approximate inverse of the $S$ block is used "
                       "in the preconditioning used in the GMRES solver. The exact definition of "
                       "this block preconditioner for the Stokes equation can be found in "
                       "\\cite{KHB12}.");

    prm.declare_entry ("Number of cheap Stokes solver steps", "200",
                       Patterns::Integer(0),
                       "As explained in the paper that describes ASPECT (Kronbichler, Heister, and Bangerth, "
                       "2012, see \\cite{KHB12}) we first try to solve the Stokes system in every "
                       "time step using a GMRES iteration with a poor but cheap "
                       "preconditioner. By default, we try whether we can converge the GMRES "
                       "solver in 200 such iterations before deciding that we need a better "
                       "preconditioner. This is sufficient for simple problems with variable "
                       "viscosity and we never need the second phase with the more expensive "
                       "preconditioner. On the other hand, for more complex problems, and in "
                       "particular for problems with strongly nonlinear viscosity, the 200 "
                       "cheap iterations don't actually do very much good and one might skip "
                       "this part right away. In that case, this parameter can be set to "
                       "zero, i.e., we immediately start with the better but more expensive "
                       "preconditioner.");

    prm.declare_entry ("Maximum number of expensive Stokes solver steps", "1000",
                       Patterns::Integer(0),
                       "This sets the maximum number of iterations used in the expensive Stokes solver. "
                       "If this value is set too low for the size of the problem, the Stokes solver will "
                       "not converge and return an error message pointing out that the user didn't allow "
                       "a sufficiently large number of iterations for the iterative solver to converge.");

    prm.declare_entry ("Temperature solver tolerance", "1e-12",
                       Patterns::Double(0,1),
                       "The relative tolerance up to which the linear system for "
                       "the temperature system gets solved. See 'linear solver "
                       "tolerance' for more details.");

    prm.declare_entry ("Composition solver tolerance", "1e-12",
                       Patterns::Double(0,1),
                       "The relative tolerance up to which the linear system for "
                       "the composition system gets solved. See 'linear solver "
                       "tolerance' for more details.");


    prm.enter_subsection("Formulation");
    {
      prm.declare_entry ("Formulation", "custom",
                         Patterns::Selection ("isothermal compression|custom|anelastic liquid approximation|boussinesq approximation"),
                         "Select a formulation for the basic equations. Different "
                         "published formulations are available in ASPECT (see the list of "
                         "possible values for this parameter in the manual for available options). "
                         "Two ASPECT specific options are\n"
                         "\\begin{enumerate}\n"
                         "  \\item `isothermal compression': ASPECT's original "
                         "formulation, using the explicit compressible mass equation, "
                         "and the full density for the temperature equation.\n"
                         "  \\item `custom': A custom selection of `Mass conservation' and "
                         "`Temperature equation'.\n"
                         "\\end{enumerate}\n\n"
                         "\\note{Warning: The `custom' option is "
                         "implemented for advanced users that want full control over the "
                         "equations solved. It is possible to choose inconsistent formulations "
                         "and no error checking is performed on the consistency of the resulting "
                         "equations.}");

      prm.declare_entry ("Mass conservation", "ask material model",
                         Patterns::Selection ("incompressible|isothermal compression|"
                                              "reference density profile|implicit reference density profile|"
                                              "ask material model"),
                         "Possible approximations for the density derivatives in the mass "
                         "conservation equation. Note that this parameter is only evaluated "
                         "if `Formulation' is set to `custom'. Other formulations ignore "
                         "the value of this parameter.");
      prm.declare_entry ("Temperature equation", "real density",
                         Patterns::Selection ("real density|reference density profile"),
                         "Possible approximations for the density in the temperature equation. "
                         "Possible approximations are `real density' and `reference density profile'. "
                         "Note that this parameter is only evaluated "
                         "if `Formulation' is set to `custom'. Other formulations ignore "
                         "the value of this parameter.");
    }
    prm.leave_subsection();

    // next declare parameters that pertain to the equations to be
    // solved, along with boundary conditions etc. note that at this
    // point we do not know yet which geometry model we will use, so
    // we do not know which symbolic names will be valid to address individual
    // parts of the boundary. we can only work around this by allowing any string
    // to indicate a boundary
    prm.enter_subsection ("Model settings");
    {
      prm.declare_entry ("Include melt transport", "false",
                         Patterns::Bool (),
                         "Whether to include the transport of melt into the model or not. If this "
                         "is set to true, two additional pressures (the fluid pressure and the "
                         "compaction pressure) will be added to the finite element. "
                         "Including melt transport in the simulation also requires that there is "
                         "one compositional field that has the name 'porosity'. This field will "
                         "be used for computing the additional pressures and the melt velocity, "
                         "and has a different advection equation than other compositional fields, "
                         "as it is effectively advected with the melt velocity.");
      prm.declare_entry ("Fixed temperature boundary indicators", "",
                         Patterns::List (Patterns::Anything()),
                         "A comma separated list of names denoting those boundaries "
                         "on which the temperature is fixed and described by the "
                         "boundary temperature object selected in its own section "
                         "of this input file. All boundary indicators used by the geometry "
                         "but not explicitly listed here will end up with no-flux "
                         "(insulating) boundary conditions."
                         "\n\n"
                         "The names of the boundaries listed here can either by "
                         "numbers (in which case they correspond to the numerical "
                         "boundary indicators assigned by the geometry object), or they "
                         "can correspond to any of the symbolic names the geometry object "
                         "may have provided for each part of the boundary. You may want "
                         "to compare this with the documentation of the geometry model you "
                         "use in your model."
                         "\n\n"
                         "This parameter only describes which boundaries have a fixed "
                         "temperature, but not what temperature should hold on these "
                         "boundaries. The latter piece of information needs to be "
                         "implemented in a plugin in the BoundaryTemperature "
                         "group, unless an existing implementation in this group "
                         "already provides what you want.");
      prm.declare_entry ("Fixed composition boundary indicators", "",
                         Patterns::List (Patterns::Anything()),
                         "A comma separated list of names denoting those boundaries "
                         "on which the composition is fixed and described by the "
                         "boundary composition object selected in its own section "
                         "of this input file. All boundary indicators used by the geometry "
                         "but not explicitly listed here will end up with no-flux "
                         "(insulating) boundary conditions."
                         "\n\n"
                         "The names of the boundaries listed here can either by "
                         "numbers (in which case they correspond to the numerical "
                         "boundary indicators assigned by the geometry object), or they "
                         "can correspond to any of the symbolic names the geometry object "
                         "may have provided for each part of the boundary. You may want "
                         "to compare this with the documentation of the geometry model you "
                         "use in your model."
                         "\n\n"
                         "This parameter only describes which boundaries have a fixed "
                         "composition, but not what composition should hold on these "
                         "boundaries. The latter piece of information needs to be "
                         "implemented in a plugin in the BoundaryComposition "
                         "group, unless an existing implementation in this group "
                         "already provides what you want.");
      prm.declare_entry ("Zero velocity boundary indicators", "",
                         Patterns::List (Patterns::Anything()),
                         "A comma separated list of names denoting those boundaries "
                         "on which the velocity is zero."
                         "\n\n"
                         "The names of the boundaries listed here can either by "
                         "numbers (in which case they correspond to the numerical "
                         "boundary indicators assigned by the geometry object), or they "
                         "can correspond to any of the symbolic names the geometry object "
                         "may have provided for each part of the boundary. You may want "
                         "to compare this with the documentation of the geometry model you "
                         "use in your model.");
      prm.declare_entry ("Tangential velocity boundary indicators", "",
                         Patterns::List (Patterns::Anything()),
                         "A comma separated list of names denoting those boundaries "
                         "on which the velocity is tangential and unrestrained, i.e., free-slip where "
                         "no external forces act to prescribe a particular tangential "
                         "velocity (although there is a force that requires the flow to "
                         "be tangential)."
                         "\n\n"
                         "The names of the boundaries listed here can either by "
                         "numbers (in which case they correspond to the numerical "
                         "boundary indicators assigned by the geometry object), or they "
                         "can correspond to any of the symbolic names the geometry object "
                         "may have provided for each part of the boundary. You may want "
                         "to compare this with the documentation of the geometry model you "
                         "use in your model.");
      prm.declare_entry ("Free surface boundary indicators", "",
                         Patterns::List (Patterns::Anything()),
                         "A comma separated list of names denoting those boundaries "
                         "where there is a free surface. Set to nothing to disable all "
                         "free surface computations."
                         "\n\n"
                         "The names of the boundaries listed here can either by "
                         "numbers (in which case they correspond to the numerical "
                         "boundary indicators assigned by the geometry object), or they "
                         "can correspond to any of the symbolic names the geometry object "
                         "may have provided for each part of the boundary. You may want "
                         "to compare this with the documentation of the geometry model you "
                         "use in your model.");
      prm.declare_entry ("Prescribed velocity boundary indicators", "",
                         Patterns::Map (Patterns::Anything(),
                                        Patterns::Selection(BoundaryVelocity::get_names<dim>())),
                         "A comma separated list denoting those boundaries "
                         "on which the velocity is prescribed, i.e., where unknown "
                         "external forces act to prescribe a particular velocity. This is "
                         "often used to prescribe a velocity that equals that of "
                         "overlying plates."
                         "\n\n"
                         "The format of valid entries for this parameter is that of a map "
                         "given as ``key1 [selector]: value1, key2 [selector]: value2, key3: value3, ...'' where "
                         "each key must be a valid boundary indicator (which is either an "
                         "integer or the symbolic name the geometry model in use may have "
                         "provided for this part of the boundary) "
                         "and each value must be one of the currently implemented boundary "
                         "velocity models. ``selector'' is an optional string given as a subset "
                         "of the letters 'xyz' that allows you to apply the boundary conditions "
                         "only to the components listed. As an example, '1 y: function' applies "
                         "the type 'function' to the y component on boundary 1. Without a selector "
                         "it will affect all components of the velocity."
                         "\n\n"
                         "Note that the no-slip boundary condition is "
                         "a special case of the current one where the prescribed velocity "
                         "happens to be zero. It can thus be implemented by indicating that "
                         "a particular boundary is part of the ones selected "
                         "using the current parameter and using ``zero velocity'' as "
                         "the boundary values. Alternatively, you can simply list the "
                         "part of the boundary on which the velocity is to be zero with "
                         "the parameter ``Zero velocity boundary indicator'' in the "
                         "current parameter section."
                         "\n\n"
                         "Note that when ``Use years in output instead of seconds'' is set "
                         "to true, velocity should be given in m/yr. ");
      prm.declare_entry ("Prescribed traction boundary indicators", "",
                         Patterns::Map (Patterns::Anything(),
                                        Patterns::Selection(BoundaryTraction::get_names<dim>())),
                         "A comma separated list denoting those boundaries "
                         "on which a traction force is prescribed, i.e., where "
                         "known external forces act, resulting in an unknown velocity. This is "
                         "often used to model ``open'' boundaries where we only know the pressure. "
                         "This pressure then produces a force that is normal to the boundary and "
                         "proportional to the pressure."
                         "\n\n"
                         "The format of valid entries for this parameter is that of a map "
                         "given as ``key1 [selector]: value1, key2 [selector]: value2, key3: value3, ...'' where "
                         "each key must be a valid boundary indicator (which is either an "
                         "integer or the symbolic name the geometry model in use may have "
                         "provided for this part of the boundary) "
                         "and each value must be one of the currently implemented boundary "
                         "traction models. ``selector'' is an optional string given as a subset "
                         "of the letters 'xyz' that allows you to apply the boundary conditions "
                         "only to the components listed. As an example, '1 y: function' applies "
                         "the type 'function' to the y component on boundary 1. Without a selector "
                         "it will affect all components of the traction.");
      prm.declare_entry ("Remove nullspace", "",
                         Patterns::MultipleSelection("net rotation|angular momentum|"
                                                     "net translation|linear momentum|"
                                                     "net x translation|net y translation|net z translation|"
                                                     "linear x momentum|linear y momentum|linear z momentum"),
                         "Choose none, one or several from "
                         "\n\n"
                         "\\begin{itemize} \\item net rotation \\item angular momentum \\item net translation "
                         "\\item linear momentum \\item net x translation \\item net y translation "
                         "\\item net z translation \\item linear x momentum \\item linear y momentum "
                         "\\item linear z momentum \\end{itemize}"
                         "\n\n"
                         "These are a selection of operations to remove certain parts of the nullspace from "
                         "the velocity after solving. For some geometries and certain boundary conditions "
                         "the velocity field is not uniquely determined but contains free translations "
                         "and/or rotations. Depending on what you specify here, these non-determined "
                         "modes will be removed from the velocity field at the end of the Stokes solve step.\n"
                         "\n\n"
                         "The ``angular momentum'' option removes a rotation such that the net angular momentum "
                         "is zero. The ``linear * momentum'' options remove translations such that the net "
                         "momentum in the relevant direction is zero.  The ``net rotation'' option removes the "
                         "net rotation of the domain, and the ``net * translation'' options remove the "
                         "net translations in the relevant directions.  For most problems there should not be a "
                         "significant difference between the momentum and rotation/translation versions of "
                         "nullspace removal, although the momentum versions are more physically motivated. "
                         "They are equivalent for constant density simulations, and approximately equivalent "
                         "when the density variations are small."
                         "\n\n"
                         "Note that while more than one operation can be selected it only makes sense to "
                         "pick one rotational and one translational operation.");
      prm.declare_entry ("Enable additional Stokes RHS", "false",
                         Patterns::Bool (),
                         "Whether to ask the material model for additional terms for the right-hand side "
                         "of the Stokes equation. This feature is likely only used when implementing force "
                         "vectors for manufactured solution problems and requires filling additional outputs "
                         "of type AdditionalMaterialOutputsStokesRHS.");

    }
    prm.leave_subsection ();

    prm.enter_subsection ("Mesh refinement");
    {
      prm.declare_entry ("Initial global refinement", "2",
                         Patterns::Integer (0),
                         "The number of global refinement steps performed on "
                         "the initial coarse mesh, before the problem is first "
                         "solved there.");
      prm.declare_entry ("Initial adaptive refinement", "0",
                         Patterns::Integer (0),
                         "The number of adaptive refinement steps performed after "
                         "initial global refinement but while still within the first "
                         "time step.");
      prm.declare_entry ("Time steps between mesh refinement", "10",
                         Patterns::Integer (0),
                         "The number of time steps after which the mesh is to be "
                         "adapted again based on computed error indicators. If 0 "
                         "then the mesh will never be changed.");
      prm.declare_entry ("Refinement fraction", "0.3",
                         Patterns::Double(0,1),
                         "The fraction of cells with the largest error that "
                         "should be flagged for refinement.");
      prm.declare_entry ("Coarsening fraction", "0.05",
                         Patterns::Double(0,1),
                         "The fraction of cells with the smallest error that "
                         "should be flagged for coarsening.");
      prm.declare_entry ("Adapt by fraction of cells", "false",
                         Patterns::Bool(),
                         "Use fraction of the total number of cells instead of "
                         "fraction of the total error as the limit for refinement "
                         "and coarsening.");
      prm.declare_entry ("Minimum refinement level", "0",
                         Patterns::Integer (0),
                         "The minimum refinement level each cell should have, "
                         "and that can not be exceeded by coarsening. "
                         "Should not be higher than the 'Initial global refinement' "
                         "parameter.");
      prm.declare_entry ("Additional refinement times", "",
                         Patterns::List (Patterns::Double(0)),
                         "A list of times so that if the end time of a time step "
                         "is beyond this time, an additional round of mesh refinement "
                         "is triggered. This is mostly useful to make sure we "
                         "can get through the initial transient phase of a simulation "
                         "on a relatively coarse mesh, and then refine again when we "
                         "are in a time range that we are interested in and where "
                         "we would like to use a finer mesh. Units: Each element of the "
                         "list has units years if the "
                         "'Use years in output instead of seconds' parameter is set; "
                         "seconds otherwise.");
      prm.declare_entry ("Run postprocessors on initial refinement", "false",
                         Patterns::Bool (),
                         "Whether or not the postproccessors should be executed after "
                         "each of the initial adaptive refinement cycles that are run at "
                         "the start of the simulation.");
    }
    prm.leave_subsection();

    prm.enter_subsection ("Postprocess");
    {
      prm.declare_entry ("Run postprocessors on nonlinear iterations", "false",
                         Patterns::Bool (),
                         "Whether or not the postproccessors should be executed after "
                         "each of the nonlinear iterations done within one time step. "
                         "As this is mainly an option for the purposes of debugging, "
                         "it is not supported when the 'Time between graphical output' "
                         "is larger than zero, or when the postprocessor is not intended "
                         "to be run more than once per timestep.");
    }
    prm.leave_subsection();

    prm.enter_subsection ("Checkpointing");
    {
      prm.declare_entry ("Time between checkpoint", "0",
                         Patterns::Integer (0),
                         "The wall time between performing checkpoints. "
                         "If 0, will use the checkpoint step frequency instead. "
                         "Units: Seconds.");
      prm.declare_entry ("Steps between checkpoint", "0",
                         Patterns::Integer (0),
                         "The number of timesteps between performing checkpoints. "
                         "If 0 and time between checkpoint is not specified, "
                         "checkpointing will not be performed. "
                         "Units: None.");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Discretization");
    {
      prm.declare_entry ("Stokes velocity polynomial degree", "2",
                         Patterns::Integer (1),
                         "The polynomial degree to use for the velocity variables "
                         "in the Stokes system. The polynomial degree for the pressure "
                         "variable will then be one less in order to make the velocity/pressure "
                         "pair conform with the usual LBB (Babuska-Brezzi) condition. In "
                         "other words, we are using a Taylor-Hood element for the Stokes "
                         "equations and this parameter indicates the polynomial degree of it. "
                         "As an example, a value of 2 for this parameter will yield the "
                         "element $Q_2^d \times Q_1$ for the $d$ velocity components and the "
                         "pressure, respectively (unless the `Use locally conservative "
                         "discretization' parameter is set, which modifies the pressure "
                         "element). "
                         "Units: None.");
      prm.declare_entry ("Temperature polynomial degree", "2",
                         Patterns::Integer (1),
                         "The polynomial degree to use for the temperature variable. "
                         "As an example, a value of 2 for this parameter will yield "
                         "either the element $Q_2$ or $DGQ_2$ for the temperature "
                         "field, depending on whether we use a continuous or "
                         "discontinuous field. "
                         "Units: None.");
      prm.declare_entry ("Composition polynomial degree", "2",
                         Patterns::Integer (1),
                         "The polynomial degree to use for the composition variable(s). "
                         "As an example, a value of 2 for this parameter will yield "
                         "either the element $Q_2$ or $DGQ_2$ for the compositional "
                         "field(s), depending on whether we use continuous or "
                         "discontinuous field(s). "
                         "Units: None.");
      prm.declare_entry ("Use locally conservative discretization", "false",
                         Patterns::Bool (),
                         "Whether to use a Stokes discretization that is locally "
                         "conservative at the expense of a larger number of degrees "
                         "of freedom (true), or to go with a cheaper discretization "
                         "that does not locally conserve mass, although it is "
                         "globally conservative (false)."
                         "\n\n"
                         "When using a locally "
                         "conservative discretization, the finite element space for "
                         "the pressure is discontinuous between cells and is the "
                         "polynomial space $P_ {-q}$ of polynomials of degree $q$ in "
                         "each variable separately. Here, $q$ is one less than the value "
                         "given in the parameter ``Stokes velocity polynomial degree''. "
                         "As a consequence of choosing this "
                         "element, it can be shown if the medium is considered incompressible "
                         "that the computed discrete velocity "
                         "field $\\mathbf u_h$ satisfies the property $\\int_ {\\partial K} \\mathbf u_h "
                         "\\cdot \\mathbf n = 0$ for every cell $K$, i.e., for each cell inflow and "
                         "outflow exactly balance each other as one would expect for an "
                         "incompressible medium. In other words, the velocity field is locally "
                         "conservative."
                         "\n\n"
                         "On the other hand, if this parameter is "
                         "set to ``false'', then the finite element space is chosen as $Q_q$. "
                         "This choice does not yield the local conservation property but "
                         "has the advantage of requiring fewer degrees of freedom. Furthermore, "
                         "the error is generally smaller with this choice."
                         "\n\n"
                         "For an in-depth discussion of these issues and a quantitative evaluation "
                         "of the different choices, see \\cite {KHB12} .");
      prm.declare_entry ("Use discontinuous temperature discretization", "false",
                         Patterns::Bool (),
                         "Whether to use a temperature discretization that is discontinuous "
                         "as opposed to continuous. This then requires the assembly of face terms "
                         "between cells, and weak imposition of boundary terms for the temperature "
                         "field via the interior-penalty discontinuous Galerkin method.");
      prm.declare_entry ("Use discontinuous composition discretization", "false",
                         Patterns::Bool (),
                         "Whether to use a composition discretization that is discontinuous "
                         "as opposed to continuous. This then requires the assembly of face terms "
                         "between cells, and weak imposition of boundary terms for the composition "
                         "field via the discontinuous Galerkin method.");

      prm.enter_subsection ("Stabilization parameters");
      {
        prm.declare_entry ("Use artificial viscosity smoothing", "false",
                           Patterns::Bool (),
                           "If set to false, the artificial viscosity of a cell is computed and"
                           "is computed on every cell separately as discussed in \\cite{KHB12}. "
                           "If set to true, the maximum of the artificial viscosity in "
                           "the cell as well as the neighbors of the cell is computed and used "
                           "instead.");
        prm.declare_entry ("alpha", "2",
                           Patterns::Integer (1, 2),
                           "The exponent $\\alpha$ in the entropy viscosity stabilization. Valid "
                           "options are 1 or 2. The recommended setting is 2. (This parameter does "
                           "not correspond to any variable in the 2012 paper by Kronbichler, "
                           "Heister and Bangerth that describes ASPECT, see \\cite{KHB12}. "
                           "Rather, the paper always uses 2 as the exponent in the definition "
                           "of the entropy, following equation (15) of the paper. The full "
                           "approach is discussed in \\cite{GPP11}.) Note that this is not the "
                           "thermal expansion coefficient, also commonly referred to as $\\alpha$."
                           "Units: None.");
        prm.declare_entry ("cR", "0.33",
                           Patterns::Double (0),
                           "The $c_R$ factor in the entropy viscosity "
                           "stabilization. (For historical reasons, the name used here is different "
                           "from the one used in the 2012 paper by Kronbichler, "
                           "Heister and Bangerth that describes ASPECT, see \\cite{KHB12}. "
                           "This parameter corresponds "
                           "to the factor $\\alpha_E$ in the formulas following equation (15) of "
                           "the paper. After further experiments, we have also chosen to use a "
                           "different value than described there.) Units: None.");
        prm.declare_entry ("beta", "0.078",
                           Patterns::Double (0),
                           "The $\\beta$ factor in the artificial viscosity "
                           "stabilization. An appropriate value for 2d is 0.078 "
                           "and 0.117 for 3d. (For historical reasons, the name used here is different "
                           "from the one used in the 2012 paper by Kronbichler, "
                           "Heister and Bangerth that describes ASPECT, see \\cite{KHB12}. "
                           "This parameter corresponds "
                           "to the factor $\\alpha_\\text {max}$ in the formulas following equation (15) of "
                           "the paper. After further experiments, we have also chosen to use a "
                           "different value than described there: It can be chosen as stated there for "
                           "uniformly refined meshes, but it needs to be chosen larger if the mesh has "
                           "cells that are not squares or cubes.) Units: None.");
        prm.declare_entry ("gamma", "0.0",
                           Patterns::Double (0),
                           "The strain rate scaling factor in the artificial viscosity "
                           "stabilization. This parameter determines how much the strain rate (in addition "
                           "to the velocity) should influence the stabilization. (This parameter does "
                           "not correspond to any variable in the 2012 paper by Kronbichler, "
                           "Heister and Bangerth that describes ASPECT, see \\cite{KHB12}. "
                           "Rather, the paper always uses "
                           "0, i.e. they specify the maximum dissipation $\\nu_h^\\text{max}$ as "
                           "$\\nu_h^\\text{max}\\vert_K = \\alpha_\\text{max} h_K \\|\\mathbf u\\|_{\\infty,K}$. "
                           "Here, we use "
                           "$\\|\\lvert\\mathbf u\\rvert + \\gamma h_K \\lvert\\varepsilon (\\mathbf u)\\rvert\\|_{\\infty,K}$ "
                           "instead of $\\|\\mathbf u\\|_{\\infty,K}$. "
                           "Units: None.");
        prm.declare_entry ("Discontinuous penalty", "10",
                           Patterns::Double (0),
                           "The value used to penalize discontinuities in the discontinuous Galerkin "
                           "method. This is used only for the temperature field, and not for the composition "
                           "field, as pure advection does not use the interior penalty method. This "
                           "is largely empirically decided -- it must be large enough to ensure "
                           "the bilinear form is coercive, but not so large as to penalize "
                           "discontinuity at all costs.");
        prm.declare_entry ("Use limiter for discontinuous temperature solution", "false",
                           Patterns::Bool (),
                           "Whether to apply the bound preserving limiter as a correction after computing "
                           "the discontinous temperature solution. Currently we apply this only to the "
                           "temperature solution if the 'Global temperature maximum' and "
                           "'Global temperature minimum' are already defined in the .prm file. "
                           "This limiter keeps the discontinuous solution in the range given by "
                           "'Global temperature maximum' and 'Global temperature minimum'.");
        prm.declare_entry ("Use limiter for discontinuous composition solution", "false",
                           Patterns::Bool (),
                           "Whether to apply the bound preserving limiter as a correction after having "
                           "the discontinous composition solution. Currently we apply this only to the "
                           "compositional solution if the 'Global composition maximum' and "
                           "'Global composition minimum' are already defined in the .prm file. "
                           "This limiter keeps the discontinuous solution in the range given by "
                           "Global composition maximum' and 'Global composition minimum'.");
        prm.declare_entry ("Global temperature maximum",
                           boost::lexical_cast<std::string>(std::numeric_limits<double>::max()),
                           Patterns::Double (),
                           "The maximum global temperature value that will be used in the bound preserving "
                           "limiter for the discontinuous solutions from temperature advection fields.");
        prm.declare_entry ("Global temperature minimum",
                           boost::lexical_cast<std::string>(-std::numeric_limits<double>::max()),
                           Patterns::Double (),
                           "The minimum global temperature value that will be used in the bound preserving "
                           "limiter for the discontinuous solutions from temperature advection fields.");
        prm.declare_entry ("Global composition maximum",
                           boost::lexical_cast<std::string>(std::numeric_limits<double>::max()),
                           Patterns::List(Patterns::Double ()),
                           "The maximum global composition values that will be used in the bound preserving "
                           "limiter for the discontinuous solutions from composition advection fields. "
                           "The number of the input 'Global composition maximum' values seperated by ',' has to be "
                           "the same as the number of the compositional fileds");
        prm.declare_entry ("Global composition minimum",
                           boost::lexical_cast<std::string>(-std::numeric_limits<double>::max()),
                           Patterns::List(Patterns::Double ()),
                           "The minimum global composition value that will be used in the bound preserving "
                           "limiter for the discontinuous solutions from composition advection fields. "
                           "The number of the input 'Global composition minimum' values seperated by ',' has to be "
                           "the same as the number of the compositional fileds");
      }
      prm.leave_subsection ();
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Compositional fields");
    {
      prm.declare_entry ("Number of fields", "0",
                         Patterns::Integer (0),
                         "The number of fields that will be advected along with the flow field, excluding "
                         "velocity, pressure and temperature.");
      prm.declare_entry ("Names of fields", "",
                         Patterns::List(Patterns::Anything()),
                         "A user-defined name for each of the compositional fields requested.");
      prm.declare_entry ("Compositional field methods", "",
                         Patterns::List (Patterns::Selection("field|particles")),
                         "A comma separated list denoting the solution method of each "
                         "compositional field. Each entry of the list must be "
                         "one of the currently implemented field types: "
                         "``field'', or ``particles''.");
      prm.declare_entry ("Mapped particle properties", "",
                         Patterns::Map (Patterns::Anything(),
                                        Patterns::Anything()),
                         "A comma separated list denoting the particle properties "
                         "that will be projected to those compositional fields that "
                         "are of the ``particles'' field type."
                         "\n\n"
                         "The format of valid entries for this parameter is that of a map "
                         "given as ``key1: value1, key2: value2 [component2], key3: value3 [component4], "
                         "...'' where each key must be a valid field name of the "
                         "``particles'' type, and each value must be one of the currently "
                         "selected particle properties. Component is a component index of "
                         "the particle property that is 0 by default, but can be set up to "
                         "n-1, where n is the number of vector components of this particle "
                         "property. The component indicator only needs to be "
                         "set if not the first component of the particle property should be "
                         "mapped (e.g. the $y$-component of the velocity at the particle positions).");
      prm.declare_entry ("List of normalized fields", "",
                         Patterns::List (Patterns::Integer(0)),
                         "A list of integers smaller than or equal to the number of "
                         "compositional fields. All compositional fields in this "
                         "list will be normalized before the first timestep. "
                         "The normalization is implemented in the following way: "
                         "First, the sum of the fields to be normalized is calculated "
                         "at every point and the global maximum is determined. "
                         "Second, the compositional fields to be normalized are "
                         "divided by this maximum.");
    }
    prm.leave_subsection ();

    // Finally declare a couple of parameters related how we should
    // evaluate the material models when assembling the matrix and
    // preconditioner
    prm.enter_subsection ("Material model");
    {
      prm.declare_entry ("Material averaging", "none",
                         Patterns::Selection(MaterialModel::MaterialAveraging::
                                             get_averaging_operation_names()),
                         "Whether or not (and in the first case, how) to do any averaging of "
                         "material model output data when constructing the linear systems "
                         "for velocity/pressure, temperature, and compositions in each "
                         "time step, as well as their corresponding preconditioners."
                         "\n\n"
                         "Possible choices: " + MaterialModel::MaterialAveraging::
                         get_averaging_operation_names()
                         +
                         "\n\n"
                         "The process of averaging, and where it may be used, is "
                         "discussed in more detail in "
                         "Section~\\ref{sec:sinker-with-averaging}."
                         "\n\n"
                         "More averaging schemes are available in the averaging material "
                         "model. This material model is a ``compositing material model'' "
                         "which can be used in combination with other material models.");
    }
    prm.leave_subsection ();

    // also declare the parameters that the FreeSurfaceHandler needs
    FreeSurfaceHandler<dim>::declare_parameters (prm);

    // then, finally, let user additions that do not go through the usual
    // plugin mechanism, declare their parameters if they have subscribed
    // to the relevant signals
    SimulatorSignals<dim>::declare_additional_parameters (dim, prm);
  }



  template <int dim>
  void
  Parameters<dim>::
  parse_parameters (ParameterHandler &prm,
                    const MPI_Comm mpi_communicator)
  {
    // first, make sure that the ParameterHandler parser agrees
    // with the code in main() about the meaning of the "Dimension"
    // parameter
    AssertThrow (prm.get_integer("Dimension") == dim,
                 ExcInternalError());

    CFL_number              = prm.get_double ("CFL number");
    use_conduction_timestep = prm.get_bool ("Use conduction timestep");
    convert_to_years        = prm.get_bool ("Use years in output instead of seconds");
    timing_output_frequency = prm.get_integer ("Timing output frequency");

    maximum_time_step       = prm.get_double("Maximum time step");
    if (convert_to_years == true)
      maximum_time_step *= year_in_seconds;

    if (prm.get ("Nonlinear solver scheme") == "IMPES")
      nonlinear_solver = NonlinearSolver::IMPES;
    else if (prm.get ("Nonlinear solver scheme") == "iterated IMPES")
      nonlinear_solver = NonlinearSolver::iterated_IMPES;
    else if (prm.get ("Nonlinear solver scheme") == "iterated Stokes")
      nonlinear_solver = NonlinearSolver::iterated_Stokes;
    else if (prm.get ("Nonlinear solver scheme") == "Stokes only")
      nonlinear_solver = NonlinearSolver::Stokes_only;
    else if (prm.get ("Nonlinear solver scheme") == "Advection only")
      nonlinear_solver = NonlinearSolver::Advection_only;
    else
      AssertThrow (false, ExcNotImplemented());

    nonlinear_tolerance = prm.get_double("Nonlinear solver tolerance");

    max_nonlinear_iterations = prm.get_integer ("Max nonlinear iterations");
    max_nonlinear_iterations_in_prerefinement = prm.get_integer ("Max nonlinear iterations in pre-refinement");

    start_time              = prm.get_double ("Start time");
    if (convert_to_years == true)
      start_time *= year_in_seconds;

    output_directory        = prm.get ("Output directory");
    if (output_directory.size() == 0)
      output_directory = "./";
    else if (output_directory[output_directory.size()-1] != '/')
      output_directory += "/";

    Utilities::create_directory (output_directory,
                                 mpi_communicator,
                                 false);

    if (prm.get ("Resume computation") == "true")
      resume_computation = true;
    else if (prm.get ("Resume computation") == "false")
      resume_computation = false;
    else if (prm.get ("Resume computation") == "auto")
      {
        std::fstream check_file((output_directory+"restart.mesh").c_str());
        resume_computation = check_file.is_open();
        check_file.close();
      }
    else
      AssertThrow (false, ExcMessage ("Resume computation parameter must be either 'true', 'false', or 'auto'."));
#ifndef DEAL_II_WITH_ZLIB
    AssertThrow (resume_computation == false,
                 ExcMessage ("You need to have deal.II configured with the 'libz' "
                             "option if you want to resume a computation from a checkpoint, but deal.II "
                             "did not detect its presence when you called 'cmake'."));
#endif

    surface_pressure                = prm.get_double ("Surface pressure");
    adiabatic_surface_temperature   = prm.get_double ("Adiabatic surface temperature");
    pressure_normalization          = prm.get("Pressure normalization");

    use_direct_stokes_solver        = prm.get_bool("Use direct solver for Stokes system");
    linear_stokes_solver_tolerance  = prm.get_double ("Linear solver tolerance");
    linear_solver_A_block_tolerance = prm.get_double ("Linear solver A block tolerance");
    linear_solver_S_block_tolerance = prm.get_double ("Linear solver S block tolerance");
    n_cheap_stokes_solver_steps     = prm.get_integer ("Number of cheap Stokes solver steps");
    n_expensive_stokes_solver_steps = prm.get_integer ("Maximum number of expensive Stokes solver steps");
    temperature_solver_tolerance    = prm.get_double ("Temperature solver tolerance");
    composition_solver_tolerance    = prm.get_double ("Composition solver tolerance");

    prm.enter_subsection ("Mesh refinement");
    {
      initial_global_refinement    = prm.get_integer ("Initial global refinement");
      initial_adaptive_refinement  = prm.get_integer ("Initial adaptive refinement");

      adaptive_refinement_interval = prm.get_integer ("Time steps between mesh refinement");
      refinement_fraction          = prm.get_double ("Refinement fraction");
      coarsening_fraction          = prm.get_double ("Coarsening fraction");
      adapt_by_fraction_of_cells   = prm.get_bool ("Adapt by fraction of cells");
      min_grid_level               = prm.get_integer ("Minimum refinement level");

      AssertThrow(refinement_fraction >= 0 && coarsening_fraction >= 0,
                  ExcMessage("Refinement/coarsening fractions must be positive."));
      AssertThrow(refinement_fraction+coarsening_fraction <= 1,
                  ExcMessage("Refinement and coarsening fractions must be <= 1."));
      AssertThrow(min_grid_level <= initial_global_refinement,
                  ExcMessage("Minimum refinement level must not be larger than "
                             "Initial global refinement."));

      // extract the list of times at which additional refinement is requested
      // then sort it and convert it to seconds
      additional_refinement_times
        = Utilities::string_to_double
          (Utilities::split_string_list(prm.get ("Additional refinement times")));
      std::sort (additional_refinement_times.begin(),
                 additional_refinement_times.end());
      if (convert_to_years == true)
        for (unsigned int i=0; i<additional_refinement_times.size(); ++i)
          additional_refinement_times[i] *= year_in_seconds;

      run_postprocessors_on_initial_refinement = prm.get_bool("Run postprocessors on initial refinement");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Postprocess");
    {
      run_postprocessors_on_nonlinear_iterations = prm.get_bool("Run postprocessors on nonlinear iterations");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Formulation");
    {
      // The following options each have a set of conditions to be met in order
      // for the formulation to be consistent, however, most of
      // the information is not available at this point. Therefore, the error checking is done
      // in Simulator<dim>::check_consistency_of_formulation() after the initialization of
      // material models, heating plugins, and adiabatic conditions.
      formulation = Formulation::parse(prm.get("Formulation"));
      if (formulation == Formulation::isothermal_compression)
        {
          formulation_mass_conservation = Formulation::MassConservation::isothermal_compression;
          formulation_temperature_equation = Formulation::TemperatureEquation::real_density;
        }
      else if (formulation == Formulation::boussinesq_approximation)
        {
          formulation_mass_conservation = Formulation::MassConservation::incompressible;
          formulation_temperature_equation = Formulation::TemperatureEquation::reference_density_profile;
        }
      else if (formulation == Formulation::anelastic_liquid_approximation)
        {
          // equally possible: implicit_reference_profile
          formulation_mass_conservation = Formulation::MassConservation::reference_density_profile;
          formulation_temperature_equation = Formulation::TemperatureEquation::reference_density_profile;
        }
      else if (formulation == Formulation::custom)
        {
          formulation_mass_conservation = Formulation::MassConservation::parse(prm.get("Mass conservation"));
          formulation_temperature_equation = Formulation::TemperatureEquation::parse(prm.get("Temperature equation"));
        }
      else AssertThrow(false, ExcNotImplemented());
    }
    prm.leave_subsection ();


    prm.enter_subsection ("Model settings");
    {
      include_melt_transport = prm.get_bool ("Include melt transport");
      enable_additional_stokes_rhs = prm.get_bool ("Enable additional Stokes RHS");

      {
        nullspace_removal = NullspaceRemoval::none;
        std::vector<std::string> nullspace_names =
          Utilities::split_string_list(prm.get("Remove nullspace"));
        AssertThrow(Utilities::has_unique_entries(nullspace_names),
                    ExcMessage("The list of strings for the parameter "
                               "'Model settings/Remove nullspace' contains entries more than once. "
                               "This is not allowed. Please check your parameter file."));

        for (unsigned int i=0; i<nullspace_names.size(); ++i)
          {
            if (nullspace_names[i]=="net rotation")
              nullspace_removal = typename NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::net_rotation);
            else if (nullspace_names[i]=="angular momentum")
              nullspace_removal = typename NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::angular_momentum);
            else if (nullspace_names[i]=="net translation")
              nullspace_removal = typename NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::net_translation_x |
                                    NullspaceRemoval::net_translation_y | ( dim == 3 ?
                                                                            NullspaceRemoval::net_translation_z : 0) );
            else if (nullspace_names[i]=="net x translation")
              nullspace_removal = typename NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::net_translation_x);
            else if (nullspace_names[i]=="net y translation")
              nullspace_removal = typename NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::net_translation_y);
            else if (nullspace_names[i]=="net z translation")
              nullspace_removal = typename NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::net_translation_z);
            else if (nullspace_names[i]=="linear x momentum")
              nullspace_removal = typename       NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::linear_momentum_x);
            else if (nullspace_names[i]=="linear y momentum")
              nullspace_removal = typename       NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::linear_momentum_y);
            else if (nullspace_names[i]=="linear z momentum")
              nullspace_removal = typename       NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::linear_momentum_z);
            else if (nullspace_names[i]=="linear momentum")
              nullspace_removal = typename NullspaceRemoval::Kind(
                                    nullspace_removal | NullspaceRemoval::linear_momentum_x |
                                    NullspaceRemoval::linear_momentum_y | ( dim == 3 ?
                                                                            NullspaceRemoval::linear_momentum_z : 0) );
            else
              AssertThrow(false, ExcInternalError());
          }
      }
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Checkpointing");
    {
      checkpoint_time_secs = prm.get_integer ("Time between checkpoint");
      checkpoint_steps     = prm.get_integer ("Steps between checkpoint");

#ifndef DEAL_II_WITH_ZLIB
      AssertThrow ((checkpoint_time_secs == 0)
                   &&
                   (checkpoint_steps == 0),
                   ExcMessage ("You need to have deal.II configured with the 'libz' "
                               "option if you want to generate checkpoints, but deal.II "
                               "did not detect its presence when you called 'cmake'."));
#endif
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Discretization");
    {
      stokes_velocity_degree = prm.get_integer ("Stokes velocity polynomial degree");
      temperature_degree     = prm.get_integer ("Temperature polynomial degree");
      composition_degree     = prm.get_integer ("Composition polynomial degree");
      use_locally_conservative_discretization
        = prm.get_bool ("Use locally conservative discretization");
      use_discontinuous_temperature_discretization
        = prm.get_bool("Use discontinuous temperature discretization");
      use_discontinuous_composition_discretization
        = prm.get_bool("Use discontinuous composition discretization");
      prm.enter_subsection ("Stabilization parameters");
      {
        use_artificial_viscosity_smoothing  = prm.get_bool ("Use artificial viscosity smoothing");
        stabilization_alpha                 = prm.get_integer ("alpha");
        stabilization_c_R                   = prm.get_double ("cR");
        stabilization_beta                  = prm.get_double ("beta");
        stabilization_gamma                 = prm.get_double ("gamma");
        discontinuous_penalty               = prm.get_double ("Discontinuous penalty");
        use_limiter_for_discontinuous_temperature_solution
          = prm.get_bool("Use limiter for discontinuous temperature solution");
        use_limiter_for_discontinuous_composition_solution
          = prm.get_bool("Use limiter for discontinuous composition solution");
        global_temperature_max_preset       = prm.get_double ("Global temperature maximum");
        global_temperature_min_preset       = prm.get_double ("Global temperature minimum");
        global_composition_max_preset       = Utilities::string_to_double
                                              (Utilities::split_string_list(prm.get ("Global composition maximum")));
        global_composition_min_preset       = Utilities::string_to_double
                                              (Utilities::split_string_list(prm.get ("Global composition minimum")));
      }
      prm.leave_subsection ();

      AssertThrow (use_locally_conservative_discretization ||
                   (stokes_velocity_degree > 1),
                   ExcMessage ("The polynomial degree for the velocity field "
                               "specified in the 'Stokes velocity polynomial degree' "
                               "parameter must be at least 2, unless you are using "
                               "a locally conservative discretization as specified by the "
                               "'Use locally conservative discretization' parameter. "
                               "This is because in the former case, the pressure element "
                               "is of one degree lower and continuous, and if you selected "
                               "a linear element for the velocity, you'd need a continuous "
                               "element of degree zero for the pressure, which does not exist."))

      if (include_melt_transport)
        {
          // The additional terms in the temperature systems have not been ported
          // to the DG formulation:
          AssertThrow(!use_discontinuous_temperature_discretization
                      && !use_discontinuous_composition_discretization,
                      ExcMessage ("Using discontinuous elements for temperature "
                                  "or composition in models with melt transport is currently not implemented."));
          // We can not have a DG p_f. While it would be possible to use a
          // discontinuous p_c, this is not tested, so we disable it for now.
          AssertThrow(!use_locally_conservative_discretization,
                      ExcMessage ("Discontinuous elements for the pressure "
                                  "in models with melt transport are not supported"));
        }
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Compositional fields");
    {
      n_compositional_fields = prm.get_integer ("Number of fields");
      if (include_melt_transport && (n_compositional_fields == 0))
        {
          AssertThrow (false,
                       ExcMessage ("If melt transport is included in the model, "
                                   "there has to be at least one compositional field."));
        }

      names_of_compositional_fields = Utilities::split_string_list (prm.get("Names of fields"));
      AssertThrow ((names_of_compositional_fields.size() == 0) ||
                   (names_of_compositional_fields.size() == n_compositional_fields),
                   ExcMessage ("The length of the list of names for the compositional "
                               "fields needs to either be empty or have length equal to "
                               "the number of compositional fields."));

      // check that the names use only allowed characters, are not empty strings and are unique
      for (unsigned int i=0; i<names_of_compositional_fields.size(); ++i)
        {
          Assert (names_of_compositional_fields[i].find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                                                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                                     "0123456789_") == std::string::npos,
                  ExcMessage("Invalid character in field " + names_of_compositional_fields[i] + ". "
                             "Names of compositional fields should consist of a "
                             "combination of letters, numbers and underscores."));
          Assert (names_of_compositional_fields[i].size() > 0,
                  ExcMessage("Invalid name of field " + names_of_compositional_fields[i] + ". "
                             "Names of compositional fields need to be non-empty."));
          for (unsigned int j=0; j<i; ++j)
            Assert (names_of_compositional_fields[i] != names_of_compositional_fields[j],
                    ExcMessage("Names of compositional fields have to be unique! " + names_of_compositional_fields[i] +
                               " is used more than once."));
        }

      // default names if list is empty
      if (names_of_compositional_fields.size() == 0)
        for (unsigned int i=0; i<n_compositional_fields; ++i)
          names_of_compositional_fields.push_back("C_" + Utilities::int_to_string(i+1));

      // if we want to solve the melt transport equations, check that one of the fields
      // has the name porosity
      if (include_melt_transport && std::find(names_of_compositional_fields.begin(),
                                              names_of_compositional_fields.end(), "porosity")
          == names_of_compositional_fields.end())
        {
          AssertThrow (false, ExcMessage ("If melt transport is included in the model, "
                                          "there has to be at least one compositional field "
                                          "with the name 'porosity'."));
        }

      const std::vector<int> n_normalized_fields = Utilities::string_to_int
                                                   (Utilities::split_string_list(prm.get ("List of normalized fields")));
      normalized_fields = std::vector<unsigned int> (n_normalized_fields.begin(),
                                                     n_normalized_fields.end());

      AssertThrow (normalized_fields.size() <= n_compositional_fields,
                   ExcMessage("Invalid input parameter file: Too many entries in List of normalized fields"));

      // global_composition_max_preset.size() and global_composition_min_preset.size() are obtained early than
      // n_compositional_fields. Therefore, we can only check if their sizes are the same here.
      if (use_limiter_for_discontinuous_temperature_solution
          || use_limiter_for_discontinuous_composition_solution)
        AssertThrow ((global_composition_max_preset.size() == (n_compositional_fields)
                      && global_composition_min_preset.size() == (n_compositional_fields)),
                     ExcMessage ("The number of multiple 'Global composition maximum' values "
                                 "and the number of multiple 'Global composition minimum' values "
                                 "have to be the same as the total number of compositional fields"));

      std::vector<std::string> x_compositional_field_methods
        = Utilities::split_string_list
          (prm.get ("Compositional field methods"));

      AssertThrow ((x_compositional_field_methods.size() == 0) ||
                   (x_compositional_field_methods.size() == 1) ||
                   (x_compositional_field_methods.size() == n_compositional_fields),
                   ExcMessage ("The length of the list of names for the field method of compositional "
                               "fields needs to be empty, or have one entry, or have a length equal to "
                               "the number of compositional fields."));

      // If no method is specified set the default, which is solve every composition
      // by a continuous field method
      if (x_compositional_field_methods.size() == 0)
        x_compositional_field_methods = std::vector<std::string> (n_compositional_fields,"field");
      // If only one method is specified assume all fields are solved by this method
      else if (x_compositional_field_methods.size() == 1)
        x_compositional_field_methods = std::vector<std::string> (n_compositional_fields,x_compositional_field_methods[0]);


      // Parse all field methods and store them, the vector should be empty
      // since nobody should have written into it yet.
      Assert(compositional_field_methods.size() == 0,
             ExcInternalError());
      compositional_field_methods.resize(n_compositional_fields);
      for (unsigned int i = 0; i < n_compositional_fields; ++i)
        {
          if (x_compositional_field_methods[i] == "field")
            compositional_field_methods[i] = AdvectionFieldMethod::fem_field;
          else if (x_compositional_field_methods[i] == "particles")
            compositional_field_methods[i] = AdvectionFieldMethod::particles;
          else
            AssertThrow(false,ExcNotImplemented());
        }


      const std::vector<std::string> x_mapped_particle_properties
        = Utilities::split_string_list
          (prm.get ("Mapped particle properties"));

      const unsigned int number_of_particle_fields =
        std::count(compositional_field_methods.begin(),compositional_field_methods.end(),AdvectionFieldMethod::particles);

      AssertThrow ((x_mapped_particle_properties.size() == number_of_particle_fields)
                   || (x_mapped_particle_properties.size() == 0),
                   ExcMessage ("The list of names for the mapped particle property fields needs to either be empty or have a length equal to "
                               "the number of compositional fields that are interpolated from particle properties."));

      for (std::vector<std::string>::const_iterator p = x_mapped_particle_properties.begin();
           p != x_mapped_particle_properties.end(); ++p)
        {
          // each entry has the format (white space is optional):
          // <name> : <value (might have spaces)> [component]
          //
          // first tease apart the two halves
          const std::vector<std::string> split_parts = Utilities::split_string_list (*p, ':');
          AssertThrow (split_parts.size() == 2,
                       ExcMessage ("The format for mapped particle properties  "
                                   "requires that each entry has the form `"
                                   "<name of field> : <particle property> [component]', "
                                   "but there does not appear to be a colon in the entry <"
                                   + *p
                                   + ">."));

          // the easy part: get the name of the compositional field
          const std::string key = split_parts[0];

          // check that the names used are actually names of fields,
          // are solved by particles, and are unique in this list
          std::vector<std::string>::iterator field_name_iterator = std::find(names_of_compositional_fields.begin(),
                                                                             names_of_compositional_fields.end(), key);
          AssertThrow (field_name_iterator
                       != names_of_compositional_fields.end(),
                       ExcMessage ("Name of field <" + key +
                                   "> appears in the parameter "
                                   "<Compositional fields/Mapped particle properties>, but "
                                   "there is no field with this name."));

          const unsigned int compositional_field_index = std::distance(names_of_compositional_fields.begin(),
                                                                       field_name_iterator);

          AssertThrow (compositional_field_methods[compositional_field_index]
                       == AdvectionFieldMethod::particles,
                       ExcMessage ("The field <" + key +
                                   "> appears in the parameter <Compositional fields/Mapped particle properties>, but "
                                   "is not advected by a particle method."));

          AssertThrow (std::count(names_of_compositional_fields.begin(),
                                  names_of_compositional_fields.end(), key) == 1,
                       ExcMessage ("Name of field <" + key +
                                   "> appears more than once in the parameter "
                                   "<Compositional fields/Mapped particle properties>."));

          // now for the rest. since we don't know whether there is a
          // component selector, start reading at the end and subtract
          // a number that might be a component selector
          std::string particle_property = split_parts[1];
          std::string component;
          if ((particle_property.size()>3) &&
              (particle_property[particle_property.size()-1] == ']'))
            {
              particle_property.erase (--particle_property.end());

              // this handles the (rare) case of multi digit components
              while ((particle_property[particle_property.size()-1] >= '0') &&
                     (particle_property[particle_property.size()-1] <= '9'))
                {
                  component.insert(component.begin(),particle_property[particle_property.size()-1]);
                  particle_property.erase (--particle_property.end());
                }

              AssertThrow (particle_property[particle_property.size()-1] == '[',
                           ExcMessage("Problem in parsing a component selector from the string <"
                                      + split_parts[1] + ">. A component selector has to be of the "
                                      "form [x], where x must be an unsigned integer between 0 "
                                      "and the maximum number of components of this particle property."));

              particle_property.erase (--particle_property.end());
            }

          // we've stopped reading component selectors now.
          // eat spaces that may be at the end of particle_property to get key
          while ((particle_property.size()>0) && (particle_property[particle_property.size()-1] == ' '))
            particle_property.erase (--particle_property.end());

          // finally, put it into the list
          mapped_particle_properties.insert(std::make_pair(compositional_field_index,
                                                           std::make_pair(particle_property,atoi(component.c_str()))));
        }
    }
    prm.leave_subsection ();


    // now also get the parameter related to material model averaging
    prm.enter_subsection ("Material model");
    {
      material_averaging
        = MaterialModel::MaterialAveraging::parse_averaging_operation_name
          (prm.get ("Material averaging"));
    }
    prm.leave_subsection ();


    // then, finally, let user additions that do not go through the usual
    // plugin mechanism, declare their parameters if they have subscribed
    // to the relevant signals
    SimulatorSignals<dim>::parse_additional_parameters (*this, prm);
  }



  template <int dim>
  void
  Parameters<dim>::
  parse_geometry_dependent_parameters(ParameterHandler &prm,
                                      const GeometryModel::Interface<dim> &geometry_model)
  {
    prm.enter_subsection ("Model settings");
    {
      try
        {
          const std::vector<types::boundary_id> x_fixed_temperature_boundary_indicators
            = geometry_model.translate_symbolic_boundary_names_to_ids(Utilities::split_string_list
                                                                      (prm.get ("Fixed temperature boundary indicators")));
          fixed_temperature_boundary_indicators
            = std::set<types::boundary_id> (x_fixed_temperature_boundary_indicators.begin(),
                                            x_fixed_temperature_boundary_indicators.end());
        }
      catch (const std::string &error)
        {
          AssertThrow (false, ExcMessage ("While parsing the entry <Model settings/Fixed temperature "
                                          "boundary indicators>, there was an error. Specifically, "
                                          "the conversion function complained as follows: "
                                          + error));
        }

      try
        {
          const std::vector<types::boundary_id> x_fixed_composition_boundary_indicators
            = geometry_model.translate_symbolic_boundary_names_to_ids (Utilities::split_string_list
                                                                       (prm.get ("Fixed composition boundary indicators")));
          fixed_composition_boundary_indicators
            = std::set<types::boundary_id> (x_fixed_composition_boundary_indicators.begin(),
                                            x_fixed_composition_boundary_indicators.end());
        }
      catch (const std::string &error)
        {
          AssertThrow (false, ExcMessage ("While parsing the entry <Model settings/Fixed composition "
                                          "boundary indicators>, there was an error. Specifically, "
                                          "the conversion function complained as follows: "
                                          + error));
        }

      try
        {
          const std::vector<types::boundary_id> x_zero_velocity_boundary_indicators
            = geometry_model.translate_symbolic_boundary_names_to_ids(Utilities::split_string_list
                                                                      (prm.get ("Zero velocity boundary indicators")));
          zero_velocity_boundary_indicators
            = std::set<types::boundary_id> (x_zero_velocity_boundary_indicators.begin(),
                                            x_zero_velocity_boundary_indicators.end());
        }
      catch (const std::string &error)
        {
          AssertThrow (false, ExcMessage ("While parsing the entry <Model settings/Zero velocity "
                                          "boundary indicators>, there was an error. Specifically, "
                                          "the conversion function complained as follows: "
                                          + error));
        }

      try
        {
          const std::vector<types::boundary_id> x_tangential_velocity_boundary_indicators
            = geometry_model.translate_symbolic_boundary_names_to_ids(Utilities::split_string_list
                                                                      (prm.get ("Tangential velocity boundary indicators")));
          tangential_velocity_boundary_indicators
            = std::set<types::boundary_id> (x_tangential_velocity_boundary_indicators.begin(),
                                            x_tangential_velocity_boundary_indicators.end());
        }
      catch (const std::string &error)
        {
          AssertThrow (false, ExcMessage ("While parsing the entry <Model settings/Tangential velocity "
                                          "boundary indicators>, there was an error. Specifically, "
                                          "the conversion function complained as follows: "
                                          + error));
        }

      try
        {
          const std::vector<types::boundary_id> x_free_surface_boundary_indicators
            = geometry_model.translate_symbolic_boundary_names_to_ids(Utilities::split_string_list
                                                                      (prm.get ("Free surface boundary indicators")));
          free_surface_boundary_indicators
            = std::set<types::boundary_id> (x_free_surface_boundary_indicators.begin(),
                                            x_free_surface_boundary_indicators.end());

          free_surface_enabled = !free_surface_boundary_indicators.empty();
        }
      catch (const std::string &error)
        {
          AssertThrow (false, ExcMessage ("While parsing the entry <Model settings/Free surface "
                                          "boundary indicators>, there was an error. Specifically, "
                                          "the conversion function complained as follows: "
                                          + error));
        }

      const std::vector<std::string> x_prescribed_velocity_boundary_indicators
        = Utilities::split_string_list
          (prm.get ("Prescribed velocity boundary indicators"));
      for (std::vector<std::string>::const_iterator p = x_prescribed_velocity_boundary_indicators.begin();
           p != x_prescribed_velocity_boundary_indicators.end(); ++p)
        {
          // each entry has the format (white space is optional):
          // <id> [x][y][z] : <value (might have spaces)>
          //
          // first tease apart the two halves
          const std::vector<std::string> split_parts = Utilities::split_string_list (*p, ':');
          AssertThrow (split_parts.size() == 2,
                       ExcMessage ("The format for prescribed velocity boundary indicators "
                                   "requires that each entry has the form `"
                                   "<id> [x][y][z] : <value>', but there does not "
                                   "appear to be a colon in the entry <"
                                   + *p
                                   + ">."));

          // the easy part: get the value
          const std::string value = split_parts[1];

          // now for the rest. since we don't know whether there is a
          // component selector, start reading at the end and subtracting
          // letters x, y and z
          std::string key_and_comp = split_parts[0];
          std::string comp;
          while ((key_and_comp.size()>0) &&
                 ((key_and_comp[key_and_comp.size()-1] == 'x')
                  ||
                  (key_and_comp[key_and_comp.size()-1] == 'y')
                  ||
                  ((key_and_comp[key_and_comp.size()-1] == 'z') && (dim==3))))
            {
              comp += key_and_comp[key_and_comp.size()-1];
              key_and_comp.erase (--key_and_comp.end());
            }

          // we've stopped reading component selectors now. there are three
          // possibilities:
          // - no characters are left. this means that key_and_comp only
          //   consisted of a single word that only consisted of 'x', 'y'
          //   and 'z's. then this would have been a mistake to classify
          //   as a component selector, and we better undo it
          // - the last character of key_and_comp is not a whitespace. this
          //   means that the last word in key_and_comp ended in an 'x', 'y'
          //   or 'z', but this was not meant to be a component selector.
          //   in that case, put these characters back.
          // - otherwise, we split successfully. eat spaces that may be at
          //   the end of key_and_comp to get key
          if (key_and_comp.size() == 0)
            key_and_comp.swap (comp);
          else if (key_and_comp[key_and_comp.size()-1] != ' ')
            {
              key_and_comp += comp;
              comp = "";
            }
          else
            {
              while ((key_and_comp.size()>0) && (key_and_comp[key_and_comp.size()-1] == ' '))
                key_and_comp.erase (--key_and_comp.end());
            }

          // finally, try to translate the key into a boundary_id. then
          // make sure we haven't seen it yet
          types::boundary_id boundary_id;
          try
            {
              boundary_id = geometry_model.translate_symbolic_boundary_name_to_id(key_and_comp);
            }
          catch (const std::string &error)
            {
              AssertThrow (false, ExcMessage ("While parsing the entry <Model settings/Prescribed "
                                              "velocity indicators>, there was an error. Specifically, "
                                              "the conversion function complained as follows: "
                                              + error));
            }

          AssertThrow (prescribed_velocity_boundary_indicators.find(boundary_id)
                       == prescribed_velocity_boundary_indicators.end(),
                       ExcMessage ("Boundary indicator <" + Utilities::int_to_string(boundary_id) +
                                   "> appears more than once in the list of indicators "
                                   "for nonzero velocity boundaries."));

          // finally, put it into the list
          prescribed_velocity_boundary_indicators[boundary_id] =
            std::pair<std::string,std::string>(comp,value);
        }

      const std::vector<std::string> x_prescribed_boundary_traction_indicators
        = Utilities::split_string_list
          (prm.get ("Prescribed traction boundary indicators"));
      for (std::vector<std::string>::const_iterator p = x_prescribed_boundary_traction_indicators.begin();
           p != x_prescribed_boundary_traction_indicators.end(); ++p)
        {
          // each entry has the format (white space is optional):
          // <id> [x][y][z] : <value (might have spaces)>
          //
          // first tease apart the two halves
          const std::vector<std::string> split_parts = Utilities::split_string_list (*p, ':');
          AssertThrow (split_parts.size() == 2,
                       ExcMessage ("The format for prescribed traction boundary indicators "
                                   "requires that each entry has the form `"
                                   "<id> [x][y][z] : <value>', but there does not "
                                   "appear to be a colon in the entry <"
                                   + *p
                                   + ">."));

          // the easy part: get the value
          const std::string value = split_parts[1];

          // now for the rest. since we don't know whether there is a
          // component selector, start reading at the end and subtracting
          // letters x, y and z
          std::string key_and_comp = split_parts[0];
          std::string comp;
          while ((key_and_comp.size()>0) &&
                 ((key_and_comp[key_and_comp.size()-1] == 'x')
                  ||
                  (key_and_comp[key_and_comp.size()-1] == 'y')
                  ||
                  ((key_and_comp[key_and_comp.size()-1] == 'z') && (dim==3))))
            {
              comp += key_and_comp[key_and_comp.size()-1];
              key_and_comp.erase (--key_and_comp.end());
            }

          // we've stopped reading component selectors now. there are three
          // possibilities:
          // - no characters are left. this means that key_and_comp only
          //   consisted of a single word that only consisted of 'x', 'y'
          //   and 'z's. then this would have been a mistake to classify
          //   as a component selector, and we better undo it
          // - the last character of key_and_comp is not a whitespace. this
          //   means that the last word in key_and_comp ended in an 'x', 'y'
          //   or 'z', but this was not meant to be a component selector.
          //   in that case, put these characters back.
          // - otherwise, we split successfully. eat spaces that may be at
          //   the end of key_and_comp to get key
          if (key_and_comp.size() == 0)
            key_and_comp.swap (comp);
          else if (key_and_comp[key_and_comp.size()-1] != ' ')
            {
              key_and_comp += comp;
              comp = "";
            }
          else
            {
              while ((key_and_comp.size()>0) && (key_and_comp[key_and_comp.size()-1] == ' '))
                key_and_comp.erase (--key_and_comp.end());
            }

          // finally, try to translate the key into a boundary_id. then
          // make sure we haven't seen it yet
          types::boundary_id boundary_id;
          try
            {
              boundary_id = geometry_model.translate_symbolic_boundary_name_to_id(key_and_comp);
            }
          catch (const std::string &error)
            {
              AssertThrow (false, ExcMessage ("While parsing the entry <Model settings/Prescribed "
                                              "traction indicators>, there was an error. Specifically, "
                                              "the conversion function complained as follows: "
                                              + error));
            }

          AssertThrow (prescribed_boundary_traction_indicators.find(boundary_id)
                       == prescribed_boundary_traction_indicators.end(),
                       ExcMessage ("Boundary indicator <" + Utilities::int_to_string(boundary_id) +
                                   "> appears more than once in the list of indicators "
                                   "for nonzero traction boundaries."));

          // finally, put it into the list
          prescribed_boundary_traction_indicators[boundary_id] =
            std::pair<std::string,std::string>(comp,value);
        }

    }
    prm.leave_subsection ();
  }



  template <int dim>
  void Simulator<dim>::declare_parameters (ParameterHandler &prm)
  {
    Parameters<dim>::declare_parameters (prm);
    MeltHandler<dim>::declare_parameters (prm);
    Postprocess::Manager<dim>::declare_parameters (prm);
    MeshRefinement::Manager<dim>::declare_parameters (prm);
    TerminationCriteria::Manager<dim>::declare_parameters (prm);
    MaterialModel::declare_parameters<dim> (prm);
    HeatingModel::Manager<dim>::declare_parameters (prm);
    GeometryModel::declare_parameters <dim>(prm);
    InitialTopographyModel::declare_parameters <dim>(prm);
    GravityModel::declare_parameters<dim> (prm);
    InitialTemperature::Manager<dim>::declare_parameters (prm);
    InitialComposition::Manager<dim>::declare_parameters (prm);
    PrescribedStokesSolution::declare_parameters<dim> (prm);
    BoundaryTemperature::declare_parameters<dim> (prm);
    BoundaryComposition::declare_parameters<dim> (prm);
    AdiabaticConditions::declare_parameters<dim> (prm);
    BoundaryVelocity::declare_parameters<dim> (prm);
    BoundaryTraction::declare_parameters<dim> (prm);
  }
}


// explicit instantiation of the functions we implement in this file
namespace aspect
{
#define INSTANTIATE(dim) \
  template Parameters<dim>::Parameters (ParameterHandler &prm, \
                                        MPI_Comm mpi_communicator); \
  template void Parameters<dim>::declare_parameters (ParameterHandler &prm); \
  template void Parameters<dim>::parse_parameters(ParameterHandler &prm, \
                                                  const MPI_Comm mpi_communicator); \
  template void Parameters<dim>::parse_geometry_dependent_parameters(ParameterHandler &prm, \
                                                                     const GeometryModel::Interface<dim> &geometry_model); \
  template void Simulator<dim>::declare_parameters (ParameterHandler &prm);

  ASPECT_INSTANTIATE(INSTANTIATE)
}
