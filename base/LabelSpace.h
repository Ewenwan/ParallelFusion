#ifndef LABEL_SPACE_H__
#define LABEL_SPACE_H__

#include <vector>
#include <glog/logging.h>
#include <algorithm>


//using namespace std;

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


template<class LabelType> LabelSpace<LabelType>::LabelSpace(const std::vector<LabelType> &single_labels) : num_nodes_(single_labels.size())
{
  label_space_.assign(num_nodes_, std::vector<LabelType>());
  setSingleLabels(single_labels);
}

template<class LabelType> LabelSpace<LabelType>::LabelSpace(const std::vector<std::vector<LabelType> > &label_space) : num_nodes_(label_space.size()), label_space_(label_space)
{
}

template<class LabelType> void LabelSpace<LabelType>::clear()
{
  for (int node_index = 0; node_index < num_nodes_; node_index++)
    label_space_[node_index].clear();
}

template<class LabelType> void LabelSpace<LabelType>::setSingleLabels(const std::vector<LabelType> &single_labels)
{
  //CHECK(single_labels.size() == num_nodes_) << "The number of nodes is inconsistent.";
  num_nodes_ = single_labels.size();
  for (int node_index = 0; node_index < num_nodes_; node_index++)
    label_space_[node_index] = std::vector<LabelType>(1, single_labels[node_index]);
}

template<class LabelType> LabelSpace<LabelType> &LabelSpace<LabelType>::operator += (const LabelSpace<LabelType> &rhs)
{
  //CHECK(label_space_.size() == rhs_label_space.size()) << "The number of nodes is inconsistent.";
  std::vector<std::vector<LabelType> > rhs_label_space = rhs.getLabelSpace();
  if (num_nodes_ == 0) {
    num_nodes_ = rhs_label_space.size();
    label_space_ = rhs_label_space;
  } else {
    for (int node_index = 0; node_index < num_nodes_; node_index++) {
      std::vector<LabelType> node_labels = label_space_[node_index];
      std::vector<LabelType> rhs_node_labels = rhs_label_space[node_index];
      //sort(node_labels.begin(), node_labels.end());    Maybe we assume?
      //sort(rhs_node_labels.begin(), rhs_node_labels.end());
      std::vector<LabelType> union_node_labels(node_labels.size() + rhs_node_labels.size());
      typename std::vector<LabelType>::const_iterator union_node_labels_it = set_union(node_labels.begin(), node_labels.end(), rhs_node_labels.begin(), rhs_node_labels.end(), union_node_labels.begin());
      union_node_labels.resize(union_node_labels_it - union_node_labels.begin());
      label_space_[node_index] = union_node_labels;
    }
  }
  return *this;
}

template<class LabelType> LabelSpace<LabelType> operator + (const LabelSpace<LabelType> &lhs, const LabelSpace<LabelType> &rhs)
{
  LabelSpace<LabelType> union_label_space = lhs;
  union_label_space += rhs;
  return union_label_space;
}

#endif
