#include "OpticalFlowFusionSolver.h"

#include <opencv2/highgui/highgui.hpp>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/trws/trws_trws.hxx>

#include "../base/cv_utils/cv_utils.h"
#include "TRW_S/MRFEnergy.h"
#include "OpticalFlowUtils.h"

using namespace std;
using namespace cv;
using namespace cv_utils;
using namespace ParallelFusion;

namespace flow_fusion {
  OpticalFlowFusionSolver::OpticalFlowFusionSolver(const cv::Mat &image_1, const cv::Mat &image_2) : image_1_(
													      image_1.clone()), image_2_(image_2.clone()), IMAGE_WIDTH_(image_1.cols), IMAGE_HEIGHT_(image_1.rows) {
    calcNeighborInfo();

      Mat blurred_image_1, blurred_image_2;
      GaussianBlur(image_1, blurred_image_1, cv::Size(5, 5), 0, 0);
      GaussianBlur(image_2, blurred_image_2, cv::Size(5, 5), 0, 0);
      subtract(image_1, blurred_image_1, image_1_high_freq_);
      subtract(image_2, blurred_image_2, image_2_high_freq_);
      cvtColor(image_1, image_1_gray_, CV_BGR2GRAY);
      cvtColor(image_2, image_2_gray_, CV_BGR2GRAY);
    }

  vector<double> OpticalFlowFusionSolver::readColorVec(const bool left_or_right, const int x, const int y) const {
      Vec3b color_high_freq = left_or_right ? image_1_high_freq_.at<Vec3b>(y, x) : image_2_high_freq_.at<Vec3b>(y, x);
      Vec3b color = left_or_right ? image_1_.at<Vec3b>(y, x) : image_2_.at<Vec3b>(y, x);
      uchar color_gray = left_or_right ? image_1_gray_.at<uchar>(y, x) : image_2_gray_.at<uchar>(y, x);
      vector<double> color_vec;

      for (int c = 0; c < 3; c++)
        color_vec.push_back(color_high_freq[c]);
      for (int c = 0; c < 3; c++)
        color_vec.push_back(color[c]);
      return color_vec;
    }

// vector<double> convertVec3bToVector(const Vec3b &color)
// {
//   vector<double> color_values(3);
//   for (int c = 0; c < 3; c++)
//     color_values[c] = color[c];
//   return color_values;
// }

    vector<double> OpticalFlowFusionSolver::getImageColor(const bool left_or_right, const double x,
                                                          const double y) const {
      int lower_x = min(max(static_cast<int>(floor(x)), 0), IMAGE_WIDTH_ - 1);
      int upper_x = min(max(static_cast<int>(ceil(x)), 0), IMAGE_WIDTH_ - 1);
      int lower_y = min(max(static_cast<int>(floor(y)), 0), IMAGE_HEIGHT_ - 1);
      int upper_y = min(max(static_cast<int>(ceil(y)), 0), IMAGE_HEIGHT_ - 1);
      vector<double> color_1 = readColorVec(left_or_right, lower_x, lower_y);
      vector<double> color_2 = readColorVec(left_or_right, lower_x, upper_y);
      vector<double> color_3 = readColorVec(left_or_right, upper_x, lower_y);
      vector<double> color_4 = readColorVec(left_or_right, upper_x, upper_y);
      if (lower_x == upper_x && lower_y == upper_y)
        return color_1;
      else if (lower_x == upper_x) {
        vector<double> average_color(color_1.size(), 0);
        for (int c = 0; c < color_1.size(); c++)
          average_color[c] = color_1[c] * (upper_y - y) + color_2[c] * (y - lower_y);
        return average_color;
      } else if (lower_y == upper_y) {
        vector<double> average_color(color_1.size(), 0);
        for (int c = 0; c < color_1.size(); c++)
          average_color[c] = color_1[c] * (upper_x - x) + color_3[c] * (x - lower_x);
        return average_color;
      } else {
        double area_1 = (x - lower_x) * (y - lower_y);
        double area_2 = (x - lower_x) * (upper_y - y);
        double area_3 = (upper_x - x) * (y - lower_y);
        double area_4 = (upper_x - x) * (upper_y - y);
        vector<double> average_color(color_1.size(), 0);
        for (int c = 0; c < color_1.size(); c++)
          average_color[c] = color_1[c] * area_4 + color_2[c] * area_3 + color_3[c] * area_2 + color_4[c] * area_1;
        return average_color;
      }
    }

    double OpticalFlowFusionSolver::calcDataCost(const int pixel, const pair<double, double> &flow) const {
      if (abs(flow.first) > IMAGE_WIDTH_ || abs(flow.second) > IMAGE_HEIGHT_)
        return 0;

      const int MU = 16;
      vector<double> color_1 = getImageColor(true, pixel % IMAGE_WIDTH_, pixel / IMAGE_WIDTH_);
      vector<double> color_2 = getImageColor(false, pixel % IMAGE_WIDTH_ + flow.first,
                                             pixel / IMAGE_WIDTH_ + flow.second);
      double color_distance = cv_utils::calcDistance(color_1, color_2);
      double data_cost = pow(color_distance, 2) / (pow(color_distance, 2) + pow(MU, 2));
      //double data_cost = pow(pow(color_distance / 255, 2) + 0.000001, 0.45);
      //if (data_cost > 2)
      //cout << pixel << '\t' << color_distance << '\t' << data_cost << endl;
      return data_cost;
    }

    double OpticalFlowFusionSolver::calcSmoothnessCost(const int pixel_1, const int pixel_2,
                                                       const pair<double, double> &flow_1,
                                                       const pair<double, double> &flow_2) const {
      if (abs(flow_1.first) > IMAGE_WIDTH_ || abs(flow_1.second) > IMAGE_HEIGHT_ || abs(flow_2.first) > IMAGE_WIDTH_ ||
          abs(flow_2.second) > IMAGE_HEIGHT_)
        return 0;

      const double NU = 0.2;
      double pixel_distance = sqrt(pow(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_, 2) +
                                   pow(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_, 2));
      double smoothness_cost = log(1 + pow(flow_1.first - flow_2.first, 2) / (2 * pow(NU, 2))) + log(1 + pow(flow_1.second - flow_2.second, 2) / (2 * pow(NU, 2)));
      //double smoothness_cost = pow(pow(flow_1.first - flow_2.first, 2) + pow(flow_1.second - flow_2.second, 2) + 0.000001, 0.5);

      vector<double> color_1 = getImageColor(true, pixel_1 % IMAGE_WIDTH_, pixel_1 / IMAGE_WIDTH_);
      vector<double> color_2 = getImageColor(false, pixel_2 % IMAGE_WIDTH_, pixel_2 / IMAGE_WIDTH_);
      double color_distance = cv_utils::calcDistance(color_1, color_2);
      double weight = exp(-pow(color_distance, 0.8));
      //  double color_difference = calcDistance(convertVec3bToVector(image_1_.at<Vec3b>(pixel_1 / IMAGE_WIDTH_, pixel_1 % IMAGE_WIDTH_)), convertVec3bToVector(image_1_.at<Vec3b>(pixel_2 / IMAGE_WIDTH_, pixel_2 % IMAGE_WIDTH_)));
      //double weight = 1; //0.01 * exp(-pow(color_difference / 30, 2) / 2);
      //double weight = color_difference < 30 ? 0.024 : 0.008;
      smoothness_cost *= weight;
      // double flow_distance = sqrt(pow(flow_1.first - flow_2.first, 2) + pow(flow_1.second - flow_2.second, 2));

      // if (flow_distance > IMAGE_WIDTH_ + IMAGE_HEIGHT_)
      //   return 0;
      // //cout << flow_distance << endl;
      // double smoothness_cost = log(1 + pow(flow_distance, 2) / (2 * pow(NU, 2))) / sqrt(pow(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_, 2) + pow(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_, 2));
      // if (smoothness_cost > 2)
      //   cout << pixel_1 << '\t' << pixel_2 << '\t' << flow_distance << '\t' << smoothness_cost << endl;
      return smoothness_cost;
    }

  
  void OpticalFlowFusionSolver::solve(const LABELSPACE &proposals, const ParallelFusion::SolutionType<LABELSPACE>& current_solution, ParallelFusion::SolutionType<LABELSPACE>& solution)
  {
      const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;

      typedef opengm::GraphicalModel<float, opengm::Adder> Model;
      vector<size_t> pixel_num_labels(NUM_PIXELS);
      vector<vector<pair<double, double> > > label_space_vec = proposals.getLabelSpace();

      for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
        pixel_num_labels[pixel] = label_space_vec[pixel].size();
      Model gm(opengm::DiscreteSpace<>(pixel_num_labels.begin(), pixel_num_labels.end()));

      typedef opengm::ExplicitFunction<float> ExplicitFunction;
      typedef Model::FunctionIdentifier FunctionIdentifier;


      unique_ptr<MRFEnergy < TypeGeneral> > energy_function(new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize()));
      MRFEnergy<TypeGeneral>::NodeId *nodes = new MRFEnergy<TypeGeneral>::NodeId[NUM_PIXELS];


      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        const size_t shape[] = {pixel_num_labels[pixel]};
        ExplicitFunction f(shape, shape + 1, 0);
        if (!onBorder(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_)) {
          for (int proposal_index = 0; proposal_index < pixel_num_labels[pixel]; proposal_index++) {
            if (pixel_num_labels[pixel] == 0) {
              cout << "empty proposal: " << pixel << endl;
              exit(1);
            }
            pair<double, double> label = label_space_vec[pixel][proposal_index];
            f(proposal_index) = calcDataCost(pixel, label);
          }
        }
        FunctionIdentifier id = gm.addFunction(f);
        size_t variable_index[] = {pixel};
        gm.addFactor(id, variable_index, variable_index + 1);


        vector<double> data_cost(pixel_num_labels[pixel], 0);
        if (!onBorder(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_)) {
          for (int proposal_index = 0; proposal_index < pixel_num_labels[pixel]; proposal_index++) {
            pair<double, double> label = label_space_vec[pixel][proposal_index];
            data_cost[proposal_index] = calcDataCost(pixel, label);
          }
        }
        nodes[pixel] = energy_function->AddNode(TypeGeneral::LocalSize(pixel_num_labels[pixel]),
                                                TypeGeneral::NodeData(&data_cost[0]));
      }

      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        if (onBorder(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_))
          continue;
        // vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
        // for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
        //cout << pixel_neighbor_weights_[pixel].size() << endl;
        for (map<int, double>::const_iterator neighbor_pixel_it = pixel_neighbor_weights_[pixel].begin();
             neighbor_pixel_it != pixel_neighbor_weights_[pixel].end(); neighbor_pixel_it++) {
          int neighbor_pixel = neighbor_pixel_it->first;
          if (neighbor_pixel < pixel)
            continue;
          if (onBorder(neighbor_pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_))
            continue;

          const size_t shape[] = {
                  pixel_num_labels[pixel],
                  pixel_num_labels[neighbor_pixel]
          };
          ExplicitFunction f(shape, shape + 2);
          for (int proposal_index = 0; proposal_index < pixel_num_labels[pixel]; proposal_index++) {
            for (int neighbor_proposal_index = 0;
                 neighbor_proposal_index < pixel_num_labels[neighbor_pixel]; neighbor_proposal_index++) {
              pair<double, double> label = label_space_vec[pixel][proposal_index];
              pair<double, double> neighbor_label = label_space_vec[neighbor_pixel][neighbor_proposal_index];
              f(proposal_index, neighbor_proposal_index) =
                      calcSmoothnessCost(pixel, neighbor_pixel, label, neighbor_label) *
                      max(neighbor_pixel_it->second, 0.0) * SMOOTHNESS_TERM_WEIGHT_;
            }
          }
          FunctionIdentifier id = gm.addFunction(f);
          size_t variable_indices[] = {pixel, neighbor_pixel};
          gm.addFactor(id, variable_indices, variable_indices + 2);


          vector<double> smoothness_cost(pixel_num_labels[pixel] * pixel_num_labels[neighbor_pixel], 0);
          for (int proposal_index = 0; proposal_index < pixel_num_labels[pixel]; proposal_index++) {
            for (int neighbor_proposal_index = 0;
                 neighbor_proposal_index < pixel_num_labels[neighbor_pixel]; neighbor_proposal_index++) {
              pair<double, double> label = label_space_vec[pixel][proposal_index];
              pair<double, double> neighbor_label = label_space_vec[neighbor_pixel][neighbor_proposal_index];
              smoothness_cost[proposal_index + neighbor_proposal_index * pixel_num_labels[pixel]] =
                      calcSmoothnessCost(pixel, neighbor_pixel, label, neighbor_label) *
                      max(neighbor_pixel_it->second, 0.0) * SMOOTHNESS_TERM_WEIGHT_;
            }
          }
          energy_function->AddEdge(nodes[pixel], nodes[neighbor_pixel],
                                   TypeGeneral::EdgeData(TypeGeneral::GENERAL, &smoothness_cost[0]));
        }
      }

      vector<pair<double, double> > fused_solution(NUM_PIXELS);
      if (false) {
        vector<size_t> selected_proposal_indices;
        opengm::TRWSi_Parameter<Model> parameter(30);
        opengm::TRWSi<Model, opengm::Minimizer> solver(gm, parameter);
        opengm::TRWSi<Model, opengm::Minimizer>::VerboseVisitorType verbose_visitor;
        solver.infer(verbose_visitor);
        solver.arg(selected_proposal_indices);
        cout << "energy: " << solver.value() << " lower bound: " << solver.bound() << endl;

        double energy = solver.value();

        for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
          fused_solution[pixel] = label_space_vec[pixel][selected_proposal_indices[pixel]];

	LABELSPACE solution_label_space(fused_solution);
        solution = make_pair(energy, solution_label_space);
      }

      if (true) {
        MRFEnergy<TypeGeneral>::Options options;
        options.m_iterMax = 1000;
        options.m_printIter = 200;
        options.m_printMinIter = 100;
        options.m_eps = 0.001;

        //energy->SetAutomaticOrdering();
        //energy->ZeroMessages();
        //energy->AddRandomMessages(0, 0, 0.001);

        double lower_bound;
	double energy;
        energy_function->Minimize_TRW_S(options, lower_bound, energy);
        for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
          fused_solution[pixel] = label_space_vec[pixel][energy_function->GetSolution(nodes[pixel])];

	LABELSPACE solution_label_space(fused_solution);
        solution = make_pair(energy, solution_label_space);

      }

    }


  void OpticalFlowFusionSolver::calcNeighborInfo() {
    {
      pixel_neighbor_weights_.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_, map<int, double>());
      for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
	int x = pixel % IMAGE_WIDTH_;
	int y = pixel / IMAGE_WIDTH_;
	if (x < IMAGE_WIDTH_ - 1)
	  pixel_neighbor_weights_[pixel][pixel + 1] = 1;
        if (x > 0)
	  pixel_neighbor_weights_[pixel][pixel - 1] = 1;
	if (y < IMAGE_HEIGHT_ - 1)
	  pixel_neighbor_weights_[pixel][pixel + IMAGE_WIDTH_] = 1;
        if (y > 0)
	  pixel_neighbor_weights_[pixel][pixel - IMAGE_WIDTH_] = 1;
      }
      return;
    }
    
      Mat color_image;
      cvtColor(image_1_, color_image, CV_BGR2HSV);
      pixel_neighbor_weights_.assign(IMAGE_WIDTH_ * IMAGE_HEIGHT_, map<int, double>());

      const int NEIGHBOR_WINDOW_SIZE = 3;
      const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;
      vector<vector<double> > guidance_image_values(3, vector<double>(NUM_PIXELS));
      for (int y = 0; y < IMAGE_HEIGHT_; y++) {
        for (int x = 0; x < IMAGE_WIDTH_; x++) {
          int pixel = y * IMAGE_WIDTH_ + x;
          Vec3b guidance_image_color = color_image.at<Vec3b>(y, x);
          for (int c = 0; c < 3; c++) {
            guidance_image_values[c][pixel] = 1.0 * guidance_image_color[c] / 256;
          }
        }
      }

      vector<vector<double> > guidance_image_means;
      vector<vector<double> > guidance_image_vars;
      calcWindowMeansAndVars(guidance_image_values, IMAGE_WIDTH_, IMAGE_HEIGHT_, NEIGHBOR_WINDOW_SIZE,
                             guidance_image_means, guidance_image_vars);

      double epsilon = 0.00001;
      for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
        vector<int> window_pixels; // = cv_utils::findWindowPixels(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, NEIGHBOR_WINDOW_SIZE);
        vector<vector<double> > guidance_image_var(3, vector<double>(3));
        for (int c_1 = 0; c_1 < 3; c_1++)
          for (int c_2 = 0; c_2 < 3; c_2++)
            guidance_image_var[c_1][c_2] = guidance_image_vars[c_1 * 3 + c_2][pixel] + epsilon / 9 * (c_1 == c_2);
        vector<double> guidance_image_mean(3);
        for (int c = 0; c < 3; c++)
          guidance_image_mean[c] = guidance_image_means[c][pixel];
        vector<vector<double> > guidance_image_var_inverse = calcInverse(guidance_image_var);
        for (vector<int>::const_iterator window_pixel_it = window_pixels.begin();
             window_pixel_it != window_pixels.end(); window_pixel_it++) {
          for (vector<int>::const_iterator other_window_pixel_it = window_pixels.begin();
               other_window_pixel_it != window_pixels.end(); other_window_pixel_it++) {
            if (*other_window_pixel_it <= *window_pixel_it)
              continue;
            vector<double> color_1(3);
            for (int c = 0; c < 3; c++)
              color_1[c] = guidance_image_values[c][*window_pixel_it];
            vector<double> color_2(3);
            for (int c = 0; c < 3; c++)
              color_2[c] = guidance_image_values[c][*other_window_pixel_it];

            double weight = 0;
            for (int c_1 = 0; c_1 < 3; c_1++)
              for (int c_2 = 0; c_2 < 3; c_2++)
                weight += (color_1[c_1] - guidance_image_mean[c_1]) * guidance_image_var_inverse[c_1][c_2] *
                          (color_2[c_2] - guidance_image_mean[c_2]);
            weight = (weight + 1) / pow(NEIGHBOR_WINDOW_SIZE, 4);

            if (abs(*window_pixel_it % IMAGE_WIDTH_ - *other_window_pixel_it % IMAGE_WIDTH_) <= 1 &&
                abs(*window_pixel_it / IMAGE_WIDTH_ - *other_window_pixel_it / IMAGE_WIDTH_) <= 1)
              pixel_neighbor_weights_[*window_pixel_it][*other_window_pixel_it] = 1; // += weight;
          }
        }
      }
    }

  double OpticalFlowFusionSolver::evaluateEnergy(const LABELSPACE & solution) const{
    const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;
    vector<pair<double, double> > solution_labels(NUM_PIXELS);
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      solution_labels[pixel] = solution.getLabelOfNode(pixel)[0];
    }
    
    double data_energy = 0;
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      if (onBorder(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_))
	continue;
      pair<double, double> label = solution_labels[pixel];
      data_energy += calcDataCost(pixel, label);
    }

    Mat smoothness_energy_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
    double smoothness_energy = 0;
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      if (onBorder(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_))
	continue;
      // vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
      // for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      //   if (*neighbor_pixel_it < pixel)
      //     continue;
      //   if (onBorder(*neighbor_pixel_it, IMAGE_WIDTH_, IMAGE_HEIGHT_))
      //     continue;
      //   pair<double, double> label = solution[pixel];
      //   pair<double, double> neighbor_label = solution[*neighbor_pixel_it];
      //   smoothness_energy += calcSmoothnessCost(pixel, *neighbor_pixel_it, label, neighbor_label) * SMOOTHNESS_TERM_WEIGHT_;
      //   smoothness_energy_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min(calcSmoothnessCost(pixel, *neighbor_pixel_it, label, neighbor_label) * SMOOTHNESS_TERM_WEIGHT_ / 3.0 * 255, 255.0);
      // }

      double pixel_smoothness_cost_sum = 0;
      for (map<int, double>::const_iterator neighbor_pixel_it = pixel_neighbor_weights_[pixel].begin();
	   neighbor_pixel_it != pixel_neighbor_weights_[pixel].end(); neighbor_pixel_it++) {
	int neighbor_pixel = neighbor_pixel_it->first;
	if (onBorder(neighbor_pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_))
	  continue;
	if (neighbor_pixel < pixel)
	  continue;
	pair<double, double> label = solution_labels[pixel];
	pair<double, double> neighbor_label = solution_labels[neighbor_pixel];
	double smoothness_cost = calcSmoothnessCost(pixel, neighbor_pixel, label, neighbor_label) *
	  max(neighbor_pixel_it->second, 0.0) * SMOOTHNESS_TERM_WEIGHT_;
	smoothness_energy += smoothness_cost;
	pixel_smoothness_cost_sum += smoothness_cost;
	double cost_temp = calcSmoothnessCost(pixel, neighbor_pixel, label, neighbor_label);
	// if (pixel == 231 * IMAGE_WIDTH_ + 244 && neighbor_pixel == 231 * IMAGE_WIDTH_ + 245) {
	// 	cout << cost_temp << '\t' << neighbor_pixel_it->second << endl;
	// 	exit(1);
	// }
	//if (energy_temp > 0.2)
	//cout << energy_temp << '\t' << neighbor_pixel_it->second << endl;
	//cout << calcSmoothnessCost(pixel, neighbor_pixel, label, neighbor_label) * neighbor_pixel_it->second * SMOOTHNESS_TERM_WEIGHT_ / 0.3 * 255 << endl;
	//if (calcSmoothnessCost(pixel, neighbor_pixel, label, neighbor_label) > 0.1)
	//cout << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << neighbor_pixel % IMAGE_WIDTH_ << '\t' << neighbor_pixel / IMAGE_WIDTH_ << '\t' << neighbor_pixel_it->second << endl;
	// if (cost_temp > 3) {
	// 	smoothness_energy_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min(cost_temp / 0.2 * 255, 255.0);
	// 	smoothness_energy_image.at<uchar>(neighbor_pixel / IMAGE_WIDTH_, neighbor_pixel % IMAGE_WIDTH_) = min(cost_temp / 0.2 * 255, 255.0);
	// }
      }
      //cout << pixel_smoothness_cost_sum << endl;
    }
    //  imwrite("Test/smoothness_energy_image.png", smoothness_energy_image);
    cout << "data energy: " << data_energy << '\t' << "smoothness energy: " << smoothness_energy << endl;
    return data_energy + smoothness_energy;
  }
}
