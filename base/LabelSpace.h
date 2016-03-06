#ifndef LABEL_SPACE_H__
#define LABEL_SPACE_H__

#include <vector>

//Label space management
template<class LabelType> class LabelSpace
{
 public:
 LabelSpace() : num_nodes_(0) {};
 LabelSpace(const int NUM_NODES) : num_nodes_(NUM_NODES), label_space_(NUM_NODES) {};
  LabelSpace(const std::vector<LabelType> &single_solution);
  LabelSpace(const std::vector<std::vector<LabelType> > &label_space);

  std::vector<std::vector<LabelType> > getLabelSpace() const { return label_space_; };
  void clear();
  void assign(const int NUM_NODES, const std::vector<LabelType> &node_labels = std::vector<LabelType>()) { num_nodes_ = NUM_NODES; label_space_.assign(NUM_NODES, node_labels); };
  void setSingleLabels(const std::vector<LabelType> &single_labels);
  void setLabelSpace(const std::vector<std::vector<LabelType> > label_space) { num_nodes_ = label_space.size(); label_space_ = label_space; };

  LabelSpace &operator += (const LabelSpace &rhs);
  
  friend LabelSpace operator + (const LabelSpace &lhs, const LabelSpace &rhs);
  
  
 private:
  int num_nodes_;
  std::vector<std::vector<LabelType> > label_space_;
};

#endif
