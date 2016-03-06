#ifndef FUSION_SOLVER_H__
#define FUSION_SOLVER_H__

#include <vector>

#include "LabelSpace.h"

template<class LabelType> class FusionSolver
{
 public:
  virtual std::vector<LabelType> solve(const LabelSpace<LabelType> &label_space, double &energy) const = 0;
  
};
  
#endif
