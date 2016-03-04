#include <vector>

using namespace std;

void LabelSpace::clear()
{
  for (int node_index = 0; node_index < NUM_NODES_; node_index++)
    label_space_[node_index].clear();
}

void LabelSpace::setSingleSolution(const vector<int> &labels)
{
  CHECK(labels.size() == NUM_NODES_) << "The number of nodes is inconsistent.";
  clear();
  for (int node_index = 0; node_index < NUM_NODES_; node_index++)
    label_space_[node_index].push_back(labels[node_index]);
}
