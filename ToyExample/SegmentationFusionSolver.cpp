#include "SegmentationFusionSolver.h"

#include <opencv2/highgui/highgui.hpp>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/trws/trws_trws.hxx>

#include "../base/cv_utils/cv_utils.h"

using namespace std;
using namespace cv;
using namespace cv_utils;


vector<int> SegmentationFusionSolver::solve(const LabelSpace &label_space, double &energy) const
{
  const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;
  const double SMOOTHNESS_TERM_WEIGHT = 0.2;
  
  typedef opengm::GraphicalModel<float, opengm::Adder> Model;  
  vector<size_t> pixel_num_labels(NUM_PIXELS);

  vector<vector<int> > label_space_vec = label_space.getLabelSpace();
  
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
    pixel_num_labels[pixel] = label_space_vec[pixel].size();
  Model gm(opengm::DiscreteSpace<>(pixel_num_labels.begin(), pixel_num_labels.end()));
  
  typedef opengm::ExplicitFunction<float> ExplicitFunction;
  typedef Model::FunctionIdentifier FunctionIdentifier;

  map<int, Vec3b> color_table;
  color_table[0] = Vec3b(255, 255, 255);
  color_table[1] = Vec3b(0, 0, 0);
  color_table[2] = Vec3b(0, 0, 255);
  color_table[3] = Vec3b(0, 255, 0);
  color_table[4] = Vec3b(255, 0, 0);
  
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    Vec3b color = image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
    const size_t shape[] = {label_space_vec[pixel].size()};
    ExplicitFunction f(shape, shape + 1);
    for (int proposal_index = 0; proposal_index < label_space_vec[pixel].size(); proposal_index++) {
      if (label_space_vec[pixel].size() == 0) {
	cout << "empty proposal: " << pixel << endl;
	exit(1);
      }
      int label = label_space_vec[pixel][proposal_index];
      if (label < 0 || label > 5) {
	cout << pixel << '\t' << label << endl;
	exit(1);
      }
      double color_diff = 0;
      for (int c = 0; c < 3; c++)
	color_diff += pow(1.0 / 255 * (color[c] - color_table[label][c]), 2);
      color_diff = sqrt(color_diff);
      f(proposal_index) = color_diff;
    }
    FunctionIdentifier id = gm.addFunction(f);
    size_t variable_index[] = {pixel};
    gm.addFactor(id, variable_index, variable_index + 1);
  }

  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      if (*neighbor_pixel_it < pixel)
	continue;
      
      const size_t shape[] = {
	label_space_vec[pixel].size(), 
	label_space_vec[*neighbor_pixel_it].size()
      };
      ExplicitFunction f(shape, shape + 2);
      for (int proposal_index = 0; proposal_index < label_space_vec[pixel].size(); proposal_index++) {
	for (int neighbor_proposal_index = 0; neighbor_proposal_index < label_space_vec[*neighbor_pixel_it].size(); neighbor_proposal_index++) {
	  int label = label_space_vec[pixel][proposal_index];
	  int neighbor_label = label_space_vec[*neighbor_pixel_it][neighbor_proposal_index];
	  if (label == neighbor_label)
	    f(proposal_index, neighbor_proposal_index) = 0;
	  else
	    f(proposal_index, neighbor_proposal_index) = SMOOTHNESS_TERM_WEIGHT;
	}
      }
      FunctionIdentifier id = gm.addFunction(f);
      size_t variable_indices[] = {pixel, *neighbor_pixel_it};
      gm.addFactor(id, variable_indices, variable_indices + 2);  
    }
  }
  
  vector<size_t> labels;
  opengm::TRWSi_Parameter<Model> parameter(30);
  opengm::TRWSi<Model, opengm::Minimizer> solver(gm, parameter);
  opengm::TRWSi<Model, opengm::Minimizer>::VerboseVisitorType verbose_visitor;
  solver.infer(verbose_visitor);
  
  solver.arg(labels);
  cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;

  energy = solver.value();

  vector<int> fused_solution(NUM_PIXELS);
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
    fused_solution[pixel] = label_space_vec[pixel][labels[pixel]];
  
  return fused_solution;
}
