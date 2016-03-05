#ifndef LABEL_SPACE_H__
#define LABEL_SPACE_H__

#include <vector>

//Label space management
class LabelSpace
{
 public:
 LabelSpace(const int NUM_NODES) : NUM_NODES_(NUM_NODES), label_space_(NUM_NODES) {};
  LabelSpace(const std::vector<int> &single_labels);
  LabelSpace(const std::vector<std::vector<int> > &label_space);

  std::vector<std::vector<int> > getLabelSpace() const { return label_space_; };
  void clear();
  void assign(const std::vector<int> &node_labels = std::vector<int>()) { label_space_.assign(NUM_NODES_, node_labels); };
  void setSingleLabels(const std::vector<int> &single_labels);
  void setLabelSpace(const std::vector<std::vector<int> > label_space) { label_space_ = label_space; };

  LabelSpace &operator += (const LabelSpace &rhs);

  friend LabelSpace operator + (const LabelSpace &lhs, const LabelSpace &rhs);
  
  
 private:
  const int NUM_NODES_;
  std::vector<std::vector<int> > label_space_;
};

#endif
