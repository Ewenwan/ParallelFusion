#include <vector>

class LabelSpace
{
 public:
 LabelSpace(const int NUM_NODES) : NUM_NODES_(NUM_NODES), label_space_(NUM_NODES) {};
  void clear() { label_space_.clear(); };
  void assign(std::vector<int> &labels = std::vector<int>()) { label_space_.assign(NUM_NODES, labels); };
  void setSingleSolution(const std::vector<int> &labels);
  
 private:
  const int NUM_NODES_;
  std::vector<std::vector<int> > label_space_;
}
