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


#ifndef __aspect__postprocess_visualization_heat_flux_map_h
#define __aspect__postprocess_visualization_heat_flux_map_h

#include <aspect/postprocess/visualization.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      /**
       * A postprocessor that computes the pointwise heat flux density through the boundaries.
       *
       * @ingroup Postprocessing
       */
      template <int dim>
      class HeatFluxMap
        : public CellDataVectorCreator<dim>,
          public SimulatorAccess<dim>
      {
        public:
          /**
           * Evaluate the solution for the heat flux.
           */
          virtual
          std::pair<std::string, Vector<float> *>
          execute () const;
      };
    }
  }
}


#endif
