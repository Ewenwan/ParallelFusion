#include "SegmentationProposalGenerator.h"

#include <iostream>

using namespace std;

LabelSpace SegmentationProposalGenerator::getProposal() const
{
  // const int NUM_LABELS = 5;
  // vector<int> node_labels(NUM_LABELS);
  // for (int c = 0; c < NUM_LABELS; c++)
  //   node_labels[c] = c;
  
  // return LabelSpace(vector<vector<int> >(IMAGE_WIDTH_ * IMAGE_HEIGHT_, node_labels));

  static int index = 0;
  cout << "index: " << index << endl;
  vector<int> proposal_solution = vector<int>(IMAGE_WIDTH_ * IMAGE_HEIGHT_, index);
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++)
    proposal_solution[pixel] = rand() % 5;
  
  LabelSpace proposal_label_space(proposal_solution);
  proposal_label_space += LabelSpace(current_solution_);
  index = (index + 1) % 5;
  return proposal_label_space;
}

vector<int> SegmentationProposalGenerator::getInitialSolution() const
{
  return vector<int>(IMAGE_WIDTH_ * IMAGE_HEIGHT_, 0);
}
