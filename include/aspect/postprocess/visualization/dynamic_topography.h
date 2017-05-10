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


#ifndef _aspect_postprocess_visualization_dynamic_topography_h
#define _aspect_postprocess_visualization_dynamic_topography_h

#include <aspect/postprocess/visualization.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      /**
       * A class derived from CellDataVectorCreator that takes an output
       * vector and computes a variable that represents the dynamic
       * topography. This quantity, strictly speaking, only makes sense at the
       * surface of the domain. Thus, the value is set to zero in all the
       * cells inside of the domain.
       *
       * The member functions are all implementations of those declared in the
       * base class. See there for their meaning.
       */
      template <int dim>
      class DynamicTopography
        : public CellDataVectorCreator<dim>,
          public SimulatorAccess<dim>
      {
        public:
          /**
           * Evaluate the solution for the dynamic topography.
           *
           * The function classes have to implement that want to output
           * cellwise data.
           * @return A pair of values with the following meaning: - The first
           * element provides the name by which this data should be written to
           * the output file. - The second element is a pointer to a vector
           * with one element per active cell on the current processor.
           * Elements corresponding to active cells that are either artificial
           * or ghost cells (in deal.II language, see the deal.II glossary)
           * will be ignored but must nevertheless exist in the returned
           * vector. While implementations of this function must create this
           * vector, ownership is taken over by the caller of this function
           * and the caller will take care of destroying the vector pointed
           * to.
           */
          virtual
          std::pair<std::string, Vector<float> *>
          execute () const;

          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters this class declares from the parameter file.
           */
          virtual
          void
          parse_parameters (ParameterHandler &prm);

        private:
          /**
           * A parameter that we read from the input file that denotes whether
           * we should subtract the mean topography or not.
           */
          bool subtract_mean_dyn_topography;

          /**
           * A parameter allows users to set the density value outside the surface
           */
          double density_above;

          /**
           * A parameter allows users to set the density value below the lower surface
           */
          double density_below;
      };
    }
  }
}

#endif
