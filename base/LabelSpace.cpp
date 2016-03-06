#include "LabelSpace.h"

#include <vector>
#include <glog/logging.h>
#include <algorithm>


using namespace std;

template<class LabelType> LabelSpace<LabelType>::LabelSpace(const std::vector<LabelType> &single_labels) : num_nodes_(single_labels.size())
{
  label_space_.assign(num_nodes_, vector<LabelType>());
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

template<class LabelType> void LabelSpace<LabelType>::setSingleLabels(const vector<LabelType> &single_labels)
{
  //CHECK(single_labels.size() == num_nodes_) << "The number of nodes is inconsistent.";
  num_nodes_ = single_labels.size(); 
  for (int node_index = 0; node_index < num_nodes_; node_index++)
    label_space_[node_index] = vector<LabelType>(1, single_labels[node_index]);
}

template<class LabelType> LabelSpace<LabelType> &LabelSpace<LabelType>::operator += (const LabelSpace<LabelType> &rhs)
{
  vector<vector<LabelType> > rhs_label_space = rhs.getLabelSpace();
  //CHECK(label_space_.size() == rhs_label_space.size()) << "The number of nodes is inconsistent.";
  for (int node_index = 0; node_index < num_nodes_; node_index++) {
    vector<LabelType> node_labels = label_space_[node_index];
    vector<LabelType> rhs_node_labels = rhs_label_space[node_index];
    //sort(node_labels.begin(), node_labels.end());    Maybe we assume?
    //sort(rhs_node_labels.begin(), rhs_node_labels.end());
    vector<LabelType> union_node_labels(node_labels.size() + rhs_node_labels.size());
    typename vector<LabelType>::const_iterator union_node_labels_it = set_union(node_labels.begin(), node_labels.end(), rhs_node_labels.begin(), rhs_node_labels.end(), union_node_labels.begin());
    union_node_labels.resize(union_node_labels_it - union_node_labels.begin());
    label_space_[node_index] = union_node_labels;
  }
  return *this;
}

template<class LabelType> LabelSpace<LabelType> operator + (const LabelSpace<LabelType> &lhs, const LabelSpace<LabelType> &rhs)
{
  LabelSpace<LabelType> union_label_space = lhs;
  union_label_space += rhs;
  return union_label_space;
}
