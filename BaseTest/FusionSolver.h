#ifndef FUSION_SOLVER_H__
#define FUSION_SOLVER_H__

#include <vector>

#include "LabelSpace.h"

class FusionSolver
{
 public:
  virtual std::vector<int> solve(const LabelSpace &label_space, double &energy) const = 0;
  
};
  
#endif
