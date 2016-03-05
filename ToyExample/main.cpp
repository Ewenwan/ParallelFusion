#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <map>

#include "SegmentationProposalGenerator.h"
#include "SegmentationFusionSolver.h"

using namespace std;
using namespace cv;

int main()
{
  Mat image = imread("Inputs/toy_example.png");
  SegmentationProposalGenerator proposal_generator(image);
  SegmentationFusionSolver fusion_solver(image);

  vector<int> current_solution = vector<int>(image.cols * image.rows, 0);
  for (int iteration = 0; iteration < 5; iteration++) {
    proposal_generator.setCurrentSolution(current_solution);
    LabelSpace label_space = proposal_generator.getProposal();
    double dummy;
    current_solution = fusion_solver.solve(label_space, dummy);
  }
  return 0;
}
