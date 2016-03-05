#include "LabelSpace.h"

#include <vector>
#include <glog/logging.h>
#include <algorithm>


using namespace std;

LabelSpace::LabelSpace(const std::vector<int> &single_labels) : NUM_NODES_(single_labels.size())
{
  setSingleLabels(single_labels);
}

LabelSpace::LabelSpace(const std::vector<std::vector<int> > &label_space) : NUM_NODES_(label_space.size()), label_space_(label_space)
{
}

void LabelSpace::clear()
{
  for (int node_index = 0; node_index < NUM_NODES_; node_index++)
    label_space_[node_index].clear();
}

void LabelSpace::setSingleLabels(const vector<int> &single_labels)
{
  CHECK(single_labels.size() == NUM_NODES_) << "The number of nodes is inconsistent.";
  for (int node_index = 0; node_index < NUM_NODES_; node_index++)
    label_space_[node_index] = vector<int>(1, single_labels[node_index]);
}

LabelSpace &LabelSpace::operator += (const LabelSpace &rhs)
{
  vector<vector<int> > rhs_label_space = rhs.getLabelSpace();
  CHECK(label_space_.size() == rhs_label_space.size()) << "The number of nodes is inconsistent.";
  for (int node_index = 0; node_index < NUM_NODES_; node_index++) {
    vector<int> node_labels = label_space_[node_index];
    vector<int> rhs_node_labels = rhs_label_space[node_index];
    //sort(node_labels.begin(), node_labels.end());    Maybe we assume?
    //sort(rhs_node_labels.begin(), rhs_node_labels.end());
    vector<int> union_node_labels(node_labels.size() + rhs_node_labels.size());
    vector<int>::const_iterator union_node_labels_it = set_union(node_labels.begin(), node_labels.end(), rhs_node_labels.begin(), rhs_node_labels.end(), union_node_labels.begin());
    union_node_labels.resize(union_node_labels_it - union_node_labels.begin());
    label_space_[node_index] = union_node_labels;
  }
  return *this;
}

LabelSpace operator + (const LabelSpace &lhs, const LabelSpace &rhs)
{
  LabelSpace union_label_space = lhs;
  union_label_space += rhs;
  return union_label_space;
}
