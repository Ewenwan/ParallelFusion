#include "OpticalFlowFusionSolver.h"

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

vector<double> convertVec3bToVector(const Vec3b &color)
{
  vector<double> color_values(3);
  for (int c = 0; c < 3; c++)
    color_values[c] = color[c];
  return color_values;
}

vector<double> getImageColor(const Mat image, const double x, const double y)
{
  const int IMAGE_WIDTH = image.cols;
  const int IMAGE_HEIGHT = image.rows;
  int lower_x = max(static_cast<int>(floor(x)), 0);
  int upper_x = min(static_cast<int>(ceil(x)), IMAGE_WIDTH - 1);
  int lower_y = max(static_cast<int>(floor(y)), 0);
  int upper_y = min(static_cast<int>(ceil(y)), IMAGE_HEIGHT - 1);
  vector<double> color_1 = convertVec3bToVector(image.at<Vec3b>(lower_y, lower_x));
  vector<double> color_2 = convertVec3bToVector(image.at<Vec3b>(upper_y, lower_x));
  vector<double> color_3 = convertVec3bToVector(image.at<Vec3b>(lower_y, upper_x));
  vector<double> color_4 = convertVec3bToVector(image.at<Vec3b>(upper_y, upper_x));
  if (lower_x == upper_x && lower_y == upper_y)
    return color_1;
  else if (lower_x == upper_x) {
    vector<double> average_color(3, 0);
    for (int c = 0; c < 3; c++)
      average_color[c] = color_1[c] * (upper_y - y) + color_2[c] * (y - lower_y);
    return average_color;
  } else if (lower_y == upper_y) {
    vector<double> average_color(3, 0);
    for (int c = 0; c < 3; c++)
      average_color[c] = color_1[c] * (upper_x - x) + color_3[c] * (x - lower_x);
    return average_color;
  } else {
    double area_1 = (x - lower_x) * (y - lower_y);
    double area_2 = (x - lower_x) * (upper_y - y);
    double area_3 = (upper_x - x) * (y - lower_y);
    double area_4 = (upper_x - x) * (upper_y - y);
    vector<double> average_color(3, 0);
    for (int c = 0; c < 3; c++)
      average_color[c] = color_1[c] * area_4 + color_2[c] * area_3 + color_3[c] * area_2 + color_4[c] * area_1;
    return average_color;
  }
}

double OpticalFlowFusionSolver::calcDataCost(const int pixel, const pair<double, double> &flow) const
{
  const int MU = 16;
  vector<double> color_1 = getImageColor(image_1_, pixel % IMAGE_WIDTH_, pixel / IMAGE_WIDTH_);
  vector<double> color_2 = getImageColor(image_2_, pixel % IMAGE_WIDTH_ + flow.first, pixel / IMAGE_WIDTH_ + flow.second);
  double color_distance = cv_utils::calcDistance(color_1, color_2);
  return pow(color_distance, 2) / (pow(color_distance, 2) + pow(MU, 2));
}

double OpticalFlowFusionSolver::calcSmoothnessCost(const int pixel_1, const int pixel_2, const pair<double, double> &flow_1, const pair<double, double> &flow_2) const
{
  const double NU = 0.2;
  double flow_distance = sqrt(pow(flow_1.first - flow_2.first, 2) + pow(flow_1.second - flow_2.second, 2));
  return log(1 + pow(flow_distance, 2) / (2 * pow(NU, 2)));
}

vector<pair<double, double> > OpticalFlowFusionSolver::solve(const LabelSpace<pair<double, double> > &label_space, double &energy) const
{
  const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;
  
  typedef opengm::GraphicalModel<float, opengm::Adder> Model;  
  vector<size_t> pixel_num_labels(NUM_PIXELS);
  vector<vector<pair<double, double> > > label_space_vec = label_space.getLabelSpace();
  
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
    pixel_num_labels[pixel] = label_space_vec[pixel].size();
  Model gm(opengm::DiscreteSpace<>(pixel_num_labels.begin(), pixel_num_labels.end()));
  
  typedef opengm::ExplicitFunction<float> ExplicitFunction;
  typedef Model::FunctionIdentifier FunctionIdentifier;

  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    const size_t shape[] = {pixel_num_labels[pixel]};
    ExplicitFunction f(shape, shape + 1);
    for (int proposal_index = 0; proposal_index < pixel_num_labels[pixel]; proposal_index++) {
      if (pixel_num_labels[pixel] == 0) {
        cout << "empty proposal: " << pixel << endl;
        exit(1);
      }
      pair<double, double> label = label_space_vec[pixel][proposal_index];
      f(proposal_index) = calcDataCost(pixel, label);
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
        pixel_num_labels[pixel],
        pixel_num_labels[*neighbor_pixel_it]
      };
      ExplicitFunction f(shape, shape + 2);
      for (int proposal_index = 0; proposal_index < pixel_num_labels[pixel]; proposal_index++) {
        for (int neighbor_proposal_index = 0; neighbor_proposal_index < pixel_num_labels[*neighbor_pixel_it]; neighbor_proposal_index++) {
          pair<double, double> label = label_space_vec[pixel][proposal_index];
          pair<double, double> neighbor_label = label_space_vec[*neighbor_pixel_it][neighbor_proposal_index];
	  f(proposal_index, neighbor_proposal_index) = calcSmoothnessCost(pixel, *neighbor_pixel_it, label, neighbor_label) * SMOOTHNESS_TERM_WEIGHT_;
        }
      }
      FunctionIdentifier id = gm.addFunction(f);
      size_t variable_indices[] = {pixel, *neighbor_pixel_it};
      gm.addFactor(id, variable_indices, variable_indices + 2);  
    }
  }
  
  vector<size_t> selected_proposal_indices;
  opengm::TRWSi_Parameter<Model> parameter(30);
  opengm::TRWSi<Model, opengm::Minimizer> solver(gm, parameter);
  opengm::TRWSi<Model, opengm::Minimizer>::VerboseVisitorType verbose_visitor;
  solver.infer(verbose_visitor);
  solver.arg(selected_proposal_indices);
  cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;

  energy = solver.value();

  vector<pair<double, double> > fused_solution(NUM_PIXELS);
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
    fused_solution[pixel] = label_space_vec[pixel][selected_proposal_indices[pixel]];
  
  return fused_solution;
}

double OpticalFlowFusionSolver::checkSolutionEnergy(const vector<pair<double, double> > &solution)
{
  const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;
  double energy = 0;
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    pair<double, double> label = solution[pixel];
    energy += calcDataCost(pixel, label);
  }

  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      if (*neighbor_pixel_it < pixel)
        continue;
      pair<double, double> label = solution[pixel];
      pair<double, double> neighbor_label = solution[*neighbor_pixel_it];
      energy += calcSmoothnessCost(pixel, *neighbor_pixel_it, label, neighbor_label) * SMOOTHNESS_TERM_WEIGHT_;
    }
  }
  return energy;
}
