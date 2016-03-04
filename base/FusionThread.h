#ifndef FUSION_THREAD_H__
#define FUSION_THREAD_H__

#include <vector>

class FusionThread
{
 public:
  virtual void runFusion();
  void setCurrentSolution(const std::vector<int> &current_solution) { current_solution_ = current_solution; };
  
 private:
  const std::vector<int> current_solution_;
  
};

#endif
