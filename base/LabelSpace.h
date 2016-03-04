#ifndef LABEL_SPACE_H__
#define LABEL_SPACE_H__

#include <vector>

class LabelSpace
{
 public:
 LabelSpace(const int NUM_NODES) : NUM_NODES_(NUM_NODES), label_space_(NUM_NODES) {};
  LabelSpace(const int NUM_NODES, const std::vector<int> &single_labels);
  LabelSpace(const int NUM_NODES, const std::vector<std::vector<int> > &label_space);

  std::vector<std::vector<int> > getLabelSpace() { return label_space; };
  void clear();
  void assign(std::vector<int> &node_labels = std::vector<int>()) { label_space_.assign(NUM_NODES, node_labels); };
  void setSingleLabels(const std::vector<int> &single_labels);
  void setLabelSpace(const std::vector<std::vector<int> > label_space) { label_space_ = label_space; };

  LabelSpace &operator += (const LabelSpace &rhs);
  
 private:
  const int NUM_NODES_;
  std::vector<std::vector<int> > label_space_;
}

friend LabelSpace oeprator + (const LabelSpace &lhs, const LabelSpace &rhs);

#endif
