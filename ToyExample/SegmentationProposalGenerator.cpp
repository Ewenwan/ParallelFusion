#include "SegmentationProposalGenerator.h"

#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

LabelSpace<int> SegmentationProposalGenerator::getProposal() const
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
  
  LabelSpace<int> proposal_label_space(proposal_solution);
  proposal_label_space += LabelSpace<int>(current_solution_);
  index = (index + 1) % 5;
  return proposal_label_space;
}

vector<int> SegmentationProposalGenerator::getInitialSolution() const
{
  return vector<int>(IMAGE_WIDTH_ * IMAGE_HEIGHT_, 0);
}

void SegmentationProposalGenerator::writeSolution(const std::vector<int> &solution, const int iteration, const int thread_index) const
{
  map<int, Vec3b> color_table;
  color_table[0] = Vec3b(255, 255, 255);
  color_table[1] = Vec3b(0, 0, 0);
  color_table[2] = Vec3b(0, 0, 255);
  color_table[3] = Vec3b(0, 255, 0);
  color_table[4] = Vec3b(255, 0, 0);

  const int IMAGE_WIDTH = 300;
  const int IMAGE_HEIGHT = 300;
  const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
  Mat solution_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    solution_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = color_table[solution[pixel]];
  }
  imwrite("Test/solution_image_" + to_string(iteration) + "_" + to_string(thread_index) + ".png", solution_image);
}
