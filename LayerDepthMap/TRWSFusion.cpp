//
//  TRWSFusion.cpp
//  SurfaceStereo
//
//  Created by Chen Liu on 11/7/14.
//  Copyright (c) 2014 Chen Liu. All rights reserved.
//

#include "TRWSFusion.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

using namespace cv;

TRWSFusion::TRWSFusion(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const RepresenterPenalties &penalties, const DataStatistics &statistics, const bool consider_surface_cost) : image_(image), point_cloud_(point_cloud), normals_(normals), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), NUM_PIXELS_(image.cols * image.rows), penalties_(penalties), statistics_(statistics), consider_surface_cost_(consider_surface_cost)
{
  calcBoundaryScores();
  calcColorDiffVar();
}

// TRWSFusion::TRWSFusion(TRWSFusion &solver)
//   : NUM_NODES_(solver.NUM_NODES_), IMAGE_WIDTH_(solver.IMAGE_WIDTH_), IMAGE_HEIGHT_(solver.IMAGE_HEIGHT_), NUM_PIXELS_(solver.IMAGE_WIDTH_ * solver.IMAGE_HEIGHT_), proposal_num_layers_(solver.proposal_num_layers_), NUM_LABELS_(solver.NUM_LABELS_), NUM_ITERATIONS_(solver.NUM_ITERATIONS_)
// {
    
// }

TRWSFusion::~TRWSFusion()
{
}


double TRWSFusion::calcDataCost(const int pixel, const int label)
{
  //int segment_id = segmentation_[pixel];
  double input_depth = point_cloud_[pixel * 3 + 2];
  //bool on_boundary = proposal_distance_to_boundaries_[pixel] != -1;
  //bool inside_ROI = proposal_ROI_mask_[pixel];
  vector<int> layer_labels(proposal_num_layers_);
  int label_temp = label;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels[layer_index] = label_temp % (proposal_num_segments_ + 1);
    label_temp /= (proposal_num_segments_ + 1);
  }

  int foremost_non_empty_layer_index = proposal_num_layers_;
  double foremost_non_empty_layer_depth = 0;
  Segment foremost_non_empty_segment;
  //vector<double> foremost_non_empty_rgb_mean;
  //vector<double> foremost_non_empty_normal;
  for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
    if (layer_labels[layer_index] < proposal_num_segments_) {
      foremost_non_empty_layer_index = layer_index;
      foremost_non_empty_layer_depth = proposal_segments_[layer_labels[layer_index]].getDepth(pixel);
      foremost_non_empty_segment = proposal_segments_[layer_labels[layer_index]];
      
      //foremost_non_empty_normal = proposal_segments_[layer_labels[layer_index]].getNormal();
      break;
    }
  }

  assert(foremost_non_empty_layer_index < proposal_num_layers_);
  double decrease_ratio = pow(penalties_.data_term_layer_decrease_ratio, proposal_num_layers_ - 1 - foremost_non_empty_layer_index);
  //foremost_non_empty_segment_confidence *= (1 - boundary_scores_[pixel] + 0.5);
  //foremost_non_empty_segment_confidence = 1;
  //assert(foremost_non_empty_segment_confidence <= 1 && foremost_non_empty_segment_confidence >= 0);
  //foremost_non_empty_segment_confidence = 1;
  int unary_cost = 0;
  //background empty cost
  {
    if (layer_labels[proposal_num_layers_ - 1] == proposal_num_segments_) {
      unary_cost += penalties_.huge_pen;
    }
  }
  //depth cost
  {
    //    double depth_diff = abs(foremost_non_empty_layer_depth - input_depth);
    double depth_diff = foremost_non_empty_segment.calcDistance(point_cloud_, pixel);
    double depth_diff_threshold = foremost_non_empty_layer_index == proposal_num_layers_ - 1 ? statistics_.background_depth_diff_tolerance : 0;
    double depth_diff_cost = input_depth < 0 ? 0 : (1 - foremost_non_empty_segment.getConfidence() * exp(-pow(max(depth_diff - depth_diff_threshold, 0.0), 2) / (2 * statistics_.depth_diff_var))) * penalties_.depth_inconsistency_pen;
    //double depth_diff_cost = log(2 / (1 + foremost_non_empty_segment.getConfidence() * exp(-pow(depth_diff, 2) / (2 * pow(statistics_.fitting_distance_threshold * penalties_.data_cost_depth_change_ratio, 2))))) / log(2) * penalties_.depth_inconsistency_pen;

    // int segment_type = foremost_non_empty_segment.getSegmentType();
    // int segment_type_cost_scale = segment_type == -1 ? penalties_.data_cost_non_plane_ratio : (segment_type == 0 ? 1 : penalties_.data_cost_non_plane_ratio);
    unary_cost += depth_diff_cost;

    // if (pixel == 118 * IMAGE_WIDTH_ + 101) {
    //   cout << layer_labels[foremost_non_empty_layer_index] << '\t' << foremost_non_empty_layer_depth << '\t' << input_depth << '\t' << depth_diff_cost << endl;
    //   //exit(1);
    // }
    
    int depth_conflict_cost = 0;
    double previous_depth = 0;
    for (int layer_index = foremost_non_empty_layer_index; layer_index < proposal_num_layers_; layer_index++) {
      if (layer_labels[layer_index] == proposal_num_segments_)
	continue;
      double depth = proposal_segments_[layer_labels[layer_index]].getDepth(pixel);
      if (depth < previous_depth - statistics_.depth_conflict_threshold) {
	cout << pixel << '\t' << layer_index << '\t' << layer_labels[layer_index] << '\t' << layer_labels[layer_index - 1] << '\t' << depth << '\t' << previous_depth << '\t' << proposal_segments_.size() << '\t' << proposal_num_segments_ << endl;
	exit(1);
	//	if (on_boundary == false)
	depth_conflict_cost += penalties_.huge_pen;
      } else
	previous_depth = depth;
    }
    unary_cost += depth_conflict_cost;
    if (depth_conflict_cost != 0)
      cout << "depth conflict: " << unary_cost << '\t' << depth_conflict_cost << endl;
    
    if (depth_diff_cost < 0)
      cout << "depth " << depth_diff_cost << '\t' << depth_diff << '\t' << pixel << '\t' << input_depth << '\t' << foremost_non_empty_layer_depth << '\t' << layer_labels[foremost_non_empty_layer_index] << endl;
  }
  //angle cost
  {
    if (input_depth > 0) {
      double angle = foremost_non_empty_segment.calcAngle(normals_, pixel);
      double normal_diff_cost = angle * penalties_.normal_inconsistency_pen;
      //double normal_diff_cost = (1 - foremost_non_empty_segment.getConfidence() * exp(-pow(angle, 2) / (2 * pow(statistics_.similar_angle_threshold * decrease_ratio, 2)))) * penalties_.normal_inconsistency_pen;
      unary_cost += normal_diff_cost;
      if (normal_diff_cost < 0)
	cout << "normal " << normal_diff_cost << '\t' << angle << '\t' << unary_cost << endl;
    }
  }
  
  //color inconsistency cost
  {
    //Vec3b color = image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
    Vec3f hsv_color = blurred_hsv_image_.at<Vec3f>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
    double color_likelihood = foremost_non_empty_segment.predictColorLikelihood(pixel, hsv_color);
    // if (pixel == 0) {
    //   cout << color_likelihood << '\t' << layer_labels[foremost_non_empty_layer_index] << endl;
    //   //exit(1);
    // }
    //cout << normalized_color_likelihood << endl;

    double color_inconsistency_cost = min(max(-color_likelihood, 0.0), -statistics_.pixel_fitting_color_likelihood_threshold) * penalties_.color_inconsistency_pen;
    //double color_inconsistency_cost = max(1 - foremost_non_empty_segment.getConfidence() * color_likelihood * decrease_ratio / (statistics_.fitting_color_likelihood_threshold * penalties_.data_cost_color_likelihood_ratio), 0.0) * penalties_.color_inconsistency_pen;
    unary_cost += color_inconsistency_cost;
    if (color_inconsistency_cost < -100 || color_inconsistency_cost > 10000)
      cout << "color " << hsv_color << '\t' << color_likelihood << endl;

    //cout << color_inconsistency_cost << endl;
    //cout << color_likelihood << endl;
										
    //   double color_probability = 1;
  //   for (int c = 0; c < 3; c++)
  //     color_probability *= exp(-pow((color[c] - foremost_non_empty_rgb_mean[c]) / statistics_.rgb_svar[c], 2) / 2);
  //   color_inconsistency_cost = log(2 / (1 + foremost_non_empty_segment_confidence * color_probability)) / log(2) * penalties_.color_inconsistency_pen;
  //   unary_cost += color_inconsistency_cost;
    
  //   if (layer_labels[foremost_non_empty_layer_index] == 5 && pixel == 72 * IMAGE_WIDTH_ + 73) {
  //     cout << foremost_non_empty_segment_confidence << endl;
  //     for (int c = 0; c < 3; c++)
  //   	cout << color[c] - foremost_non_empty_rgb_mean[c] << endl;
  //     cout << color_inconsistency_cost << endl;
  //   }
  }
  // //distance_2D cost
  // {
  //   double distance_ratio = foremost_non_empty_segment.calcDistanceRatio2D(pixel);
  //   double distance_2D_cost = max(distance_ratio - 1, 0.0) * penalties_.distance_2D_pen;
  //   unary_cost += distance_2D_cost;
  // }
  
  //close parallel surface cost
  // {
  //   Segment previous_segment = foremost_non_empty_segment;
  //   for (int layer_index = foremost_non_empty_layer_index + 1; layer_index < proposal_num_layers_; layer_index++) {
  //     int surface_id = layer_labels[layer_index];
  //     if (surface_id == proposal_num_segments_)
  //   	continue;
  //     bool close_parallel = checkCloseParallelSegments(previous_segment, proposal_segments_[surface_id], pixel, statistics_.fitting_distance_threshold, statistics_.parallel_angle_threshold);
  //     if (close_parallel)
  //   	unary_cost += penalties_.close_parallel_surface_pen;
  //   }
  // }

  // //layer empty cost
  // {
  //   for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
  //     int surface_id = layer_labels[layer_index];
  //     if (surface_id == proposal_num_segments_)
  // 	unary_cost += penalties_.layer_empty_pen;
  //   }
  // }
  
  //same label cost
  {
    int same_label_cost = 0;
    set<int> used_surface_ids;
    for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
      if (layer_labels[layer_index] == proposal_num_segments_)
	continue;
      if (used_surface_ids.count(layer_labels[layer_index]) > 0)
	same_label_cost += penalties_.huge_pen;
      used_surface_ids.insert(layer_labels[layer_index]);
    }
    //cout << "same label cost: "  << same_label_cost << endl;
    unary_cost += same_label_cost;
  }
  //non-plane segment cost
  {
    int segment_type = foremost_non_empty_segment.getSegmentType();
    int segment_type_cost_scale = segment_type == -1 ? 2 : (segment_type == 0 ? 0 : 1);
    //    int non_plane_segment_cost = segment_type_cost_scale * (1 - exp(-1 / (2 * pow(penalties_.data_cost_non_plane_ratio, 2)))) * penalties_.depth_inconsistency_pen;
    int non_plane_segment_cost = segment_type_cost_scale * penalties_.data_cost_non_plane_ratio * penalties_.depth_inconsistency_pen;
    //cout << "non plane segment cost: "  << non_plane_segment_cost << endl;
    unary_cost += non_plane_segment_cost;
  }
  
  // //layer occupacy cost
  // {
  //   int layer_occupacy_cost = 0;
  //   for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
  //     if (layer_labels[layer_index] == proposal_num_segments_)
  // 	continue;
  //     layer_occupacy_cost += penalties_.layer_occupacy_pen;
  //   }
  //   unary_cost += layer_occupacy_cost;
  // }
  
  // //front layer cost
  // {
  //   if (foremost_non_empty_layer_index == 0)
  //     unary_cost += penalties_.front_layer_pen;
  // }

  // if (unary_cost < 0 || unary_cost > 1000000) {
  //   cout << "unary cost: " << unary_cost << endl;
  //   exit(1);
  // }
  if (unary_cost < 0) {
    cout << "negative cost: " << pixel << '\t' << label << '\t' << unary_cost << '\t' << layer_labels[foremost_non_empty_layer_index] << endl;
    exit(1);
  }
  return unary_cost;
}

double TRWSFusion::calcSmoothnessCost(const int pixel_1, const int pixel_2, const int label_1, const int label_2)
{
  if (label_1 == label_2)
    return 0;
  vector<int> layer_labels_1(proposal_num_layers_);
  int label_temp_1 = label_1;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels_1[layer_index] = label_temp_1 % (proposal_num_segments_ + 1);
    label_temp_1 /= (proposal_num_segments_ + 1);
  }
  vector<int> layer_labels_2(proposal_num_layers_);
  int label_temp_2 = label_2;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels_2[layer_index] = label_temp_2 % (proposal_num_segments_ + 1);
    label_temp_2 /= (proposal_num_segments_ + 1);
  }
  
  double pairwise_cost = 0;
  double max_boundary_score = max(boundary_scores_[pixel_1], boundary_scores_[pixel_2]);
  bool surface_1_visible = true;
  bool surface_2_visible = true;
  for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 == surface_id_2) {
      if (surface_id_1 < proposal_num_segments_) {
	surface_1_visible = false;
	surface_2_visible = false;
	continue;
      }
    }
    if (surface_id_1 < proposal_num_segments_ && surface_id_2 < proposal_num_segments_) {
      double depth_1_1 = proposal_segments_[surface_id_1].getDepth(pixel_1);
      double depth_1_2 = proposal_segments_[surface_id_1].getDepth(pixel_2);
      double depth_2_1 = proposal_segments_[surface_id_2].getDepth(pixel_1);
      double depth_2_2 = proposal_segments_[surface_id_2].getDepth(pixel_2);

      if (depth_1_1 <= 0 || depth_1_2 <= 0 || depth_2_1 <= 0 || depth_2_2 <= 0)
	return penalties_.large_pen;
  
      double diff_1 = abs(depth_1_1 - depth_2_1);
      // if (depth_change_1 < penalties_.smoothness_pen_weight)
      //   depth_change_1 = 0;
      double diff_2 = abs(depth_1_2 - depth_2_2);
      // if (depth_change_2 < penalties_.smoothness_pen_weight)
      //   depth_change_2 = 0;
      double diff_middle = (depth_1_1 - depth_2_1) * (depth_1_2 - depth_2_2) <= 0 ? 0 : 1000000;
      double min_diff = min(min(diff_1, diff_2), diff_middle);
      //double diff_middle = (depth_1_1 - depth_2_1 + depth_1_2 - depth_2_2) / 2;
      // if (depth_change_middle < penalties_.smoothness_pen_weight)
      //   depth_change_middle = 0;
      
      double boundary_score = (surface_1_visible == true && surface_2_visible == true) ? max_boundary_score : 1;
      pairwise_cost += min(min_diff * pow(penalties_.smoothness_term_layer_decrease_ratio, proposal_num_layers_ - 1 - layer_index) / statistics_.depth_change_smoothness_threshold * penalties_.smoothness_empty_non_empty_ratio, 1.0) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
      //pairwise_cost += (1 - exp(-pow(min_diff, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2)))) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
      //boundary_score = 1;
      // double cost_1 = log(2 / (1 + boundary_score * exp(-pow(diff_1, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen;
      // double cost_2 = log(2 / (1 + boundary_score * exp(-pow(diff_2, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen;
      // double cost_middle = log(2 / (1 + boundary_score * exp(-pow(diff_middle, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen;
        
      //pairwise_cost += log(2 / (1 + boundary_score * exp(-pow(min_diff, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
      // if (abs(depth_1_1 - depth_2_1) < penalties_.smoothness_depth_change_threshold || abs(depth_1_2 - depth_2_2) < penalties_.smoothness_depth_change_threshold || abs((depth_1_1 - depth_2_1 + depth_1_2 - depth_2_2) / 2) < penalties_.smoothness_depth_change_threshold)
      // 	cout << min(min(cost_1, cost_2), cost_middle) + penalties_.smoothness_small_constant_pen << '\t' << log(2 / (1 + boundary_score * exp(-0.5))) / log(2) * penalties_.smoothness_pen_weight + penalties_.smoothness_small_constant_pen << '\t' << depth_1_1 << endl;
      
      surface_1_visible = false;
      surface_2_visible = false;

      // if (layer_index == 2 && (surface_id_1 == 6 || surface_id_2 == 6))
      //   cout << pairwise_cost << endl;

    } else if (surface_id_1 < proposal_num_segments_ || surface_id_2 < proposal_num_segments_) {
      double boundary_score = 1;
      if (surface_id_1 < proposal_num_segments_ && surface_1_visible) {
	boundary_score = max_boundary_score;
	surface_1_visible = false;
      }
      if (surface_id_2 < proposal_num_segments_ && surface_2_visible) {
	boundary_score = max_boundary_score;
	surface_2_visible = false;
      }
      pairwise_cost += penalties_.smoothness_empty_non_empty_ratio * penalties_.smoothness_pen;
      
      //pairwise_cost += (1 - boundary_score * exp(-1 / (2 * pow(penalties_.smoothness_cost_depth_change_ratio, 2)))) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
      //boundary_score = 1;
      //pairwise_cost += log(2 / (1 + boundary_score * exp(-pow((penalties_.smoothness_depth_change_threshold), 2) / (2 * statistics_.disp_residual)))) / log(2) * penalties_.smoothness_pen_weight + penalties_.smoothness_small_constant_pen;
      //pairwise_cost += log(2 / (1 + boundary_score * exp(-1 / (2 * pow(penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
      
      //cout << statistics_.depth_svar << '\t' << pairwise_cost << endl;
      //exit(1);
      //      pairwise_cost += penalties_.smoothness_between_empty_pen;
    }

    // if (pixel_1 == 11594)
    //   cout << boundary_scores_[pixel_1] << '\t' << surface_id_1 << '\t' << surface_id_2 << '\t' << pairwise_cost << endl;
  }
  //  if (front_layer_cost == penalties_.smoothness_between_empty_pen)
  //    cout << front_layer_cost + back_layer_cost << endl;
  
  // pairwise_cost *= exp(-pow(max_boundary_score, 2));

  surface_1_visible = true;
  surface_2_visible = true;
  for (int layer_index = 0; layer_index < proposal_num_layers_ - 1; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 < proposal_num_segments_) {
      if (surface_1_visible == true) {
        if (surface_id_1 != surface_id_2 && proposal_segments_[surface_id_1].calcDistanceOffset(pixel_1, pixel_2) == 1)
	  pairwise_cost += penalties_.smoothness_concave_shape_pen;
	surface_1_visible = false;
      }
      // if (surface_id_1 == 2)
      // 	cout << "why" << endl;
    }
    if (surface_id_2 < proposal_num_segments_) {
      if (surface_1_visible == true) {
	if (surface_id_1 != surface_id_2 && proposal_segments_[surface_id_2].calcDistanceOffset(pixel_2, pixel_1) == 1)
	  pairwise_cost += penalties_.smoothness_concave_shape_pen;
	surface_2_visible = false;
      }
      // if (surface_id_2 == 2)
      //   cout << "why" << endl;
    }
  }
  
  for (int layer_index = 0; layer_index < proposal_num_layers_ - 1; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    for (int other_layer_index = layer_index + 1; other_layer_index < proposal_num_layers_; other_layer_index++)
      if (layer_labels_1[layer_index] != layer_labels_2[layer_index] && layer_labels_1[layer_index] == layer_labels_2[other_layer_index] && layer_labels_1[other_layer_index] == layer_labels_2[layer_index])
	pairwise_cost += penalties_.smoothness_segment_splitted_pen;
  }
  // if (pixel_1 == 43 * IMAGE_WIDTH_ + 165 && pixel_2 == 43 * IMAGE_WIDTH_ + 166) {
  //   for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
  //     int surface_id_1 = layer_labels_1[layer_index];
  //     int surface_id_2 = layer_labels_2[layer_index];
  //     cout << surface_id_1 << '\t' << surface_id_2 << endl;
  //   }
  //   exit(1);
  // }
  
  // int visible_layer_index_1 = -1;
  // int visible_layer_index_2 = -1;
  // int visible_surface_id_1 = -1;
  // int visible_surface_id_2 = -1;
  // for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
  //   int surface_id_1 = layer_labels_1[layer_index];
  //   int surface_id_2 = layer_labels_2[layer_index];
  //   if (visible_surface_id_1 == -1 && surface_id_1 < proposal_num_segments_) {
  //     visible_surface_id_1 = surface_id_1;
  //     visible_layer_index_1 = layer_index;
  //   }
  //   if (visible_surface_id_2 == -1 && surface_id_2 < proposal_num_segments_) {
  //     visible_surface_id_2 = surface_id_2;
  //     visible_layer_index_2 = layer_index;
  //   }
  // }
  
  // if (visible_layer_index_1 > visible_layer_index_2 && visible_surface_id_1 != visible_surface_id_2 && proposal_segments_[visible_surface_id_1].checkPixelFitting(point_cloud_, normals_, pixel_1) == false && proposal_segments_[visible_surface_id_2].checkPixelFitting(point_cloud_, normals_, pixel_1) == true)
  //   pairwise_cost += penalties_.smoothness_spurious_empty_pen;
  // if (visible_layer_index_2 > visible_layer_index_1 && visible_surface_id_1 != visible_surface_id_2 && proposal_segments_[visible_surface_id_2].checkPixelFitting(point_cloud_, normals_, pixel_2) == false && proposal_segments_[visible_surface_id_1].checkPixelFitting(point_cloud_, normals_, pixel_2) == true)
  //   pairwise_cost += penalties_.smoothness_spurious_empty_pen;


  int visible_surface_1 = -1;
  int visible_surface_2 = -1;
  int visible_layer_index_1 = -1;
  int visible_layer_index_2 = -1;
  for (int layer_index = 0; layer_index < proposal_num_layers_ - 1; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 < proposal_num_segments_) {
      if (visible_surface_1 == -1) {
        visible_surface_1 = surface_id_1;
        visible_layer_index_1 = layer_index;
      }
      // if (surface_id_1 == 2)
      //        cout << "why" << endl;
    }
    if (surface_id_2 < proposal_num_segments_) {
      if (visible_surface_2 == -1) {
        visible_surface_2 = surface_id_2;
        visible_layer_index_2 = layer_index;
      }
      // if (surface_id_2 == 2)
      //   cout << "why" << endl;
    }
  }

  if (visible_surface_1 != visible_surface_2) { // && visible_layer_index_1 != visible_layer_index_2) {
    pairwise_cost += exp(-pow(calcColorDiff(pixel_1, pixel_2), 2) / (2 * color_diff_var_)) * penalties_.smoothness_boundary_pen;
  }
  
  double distance_2D = sqrt(pow(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_, 2) + pow(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_, 2));
  return pairwise_cost / distance_2D;
}

// double TRWSFusion::calcSmoothnessCostMulti(const int pixel_1, const int pixel_2, const int label_1, const int label_2)
// {
//   if (label_1 == label_2)
//     return 0;
//   vector<int> layer_labels_1(proposal_num_layers_);
//   int label_temp_1 = label_1;
//   for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
//     layer_labels_1[layer_index] = label_temp_1 % (proposal_num_segments_ + 1);
//     label_temp_1 /= (proposal_num_segments_ + 1);
//   }
//   vector<int> layer_labels_2(proposal_num_layers_);
//   int label_temp_2 = label_2;
//   for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
//     layer_labels_2[layer_index] = label_temp_2 % (proposal_num_segments_ + 1);
//     label_temp_2 /= (proposal_num_segments_ + 1);
//   }
  
//   double pairwise_cost = 0;
//   for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
//     int surface_id_1 = layer_labels_1[layer_index];
//     int surface_id_2 = layer_labels_2[layer_index];
//     if (surface_id_1 < proposal_num_segments_ && surface_id_2 < proposal_num_segments_) {
//       if (surface_id_1 != surface_id_2) {
// 	if (proposal_surface_depths_[surface_id_1][pixel_1] <= 0 || proposal_surface_depths_[surface_id_1][pixel_2] <= 0 || proposal_surface_depths_[surface_id_2][pixel_1] <= 0 || proposal_surface_depths_[surface_id_2][pixel_2] <= 0)
//           return penalties_.large_pen;
  
// 	if (false) {
// 	  double depth_1 = (proposal_surface_depths_[surface_id_1][pixel_1] + proposal_surface_depths_[surface_id_1][pixel_2]) / 2;
// 	  double depth_2 = (proposal_surface_depths_[surface_id_2][pixel_1] + proposal_surface_depths_[surface_id_2][pixel_2]) / 2;
// 	  pairwise_cost += min(max(abs(depth_1 - depth_2) - penalties_.smoothness_depth_change_threshold, 0.0), penalties_.smoothness_max_depth_change) * penalties_.smoothness_pen_weight + penalties_.smoothness_small_constant_pen;
// 	} else {
// 	  if (proposal_surface_depths_[surface_id_1][pixel_1] <= 0 || proposal_surface_depths_[surface_id_1][pixel_2] <= 0 || proposal_surface_depths_[surface_id_2][pixel_1] <= 0 || proposal_surface_depths_[surface_id_2][pixel_2] <= 0) {
//             pairwise_cost += penalties_.smoothness_dangerous_boundary_pen;
// 	    continue;
// 	  }
  
// 	  double diff_1 = proposal_surface_depths_[surface_id_1][pixel_1] - proposal_surface_depths_[surface_id_2][pixel_1];
// 	  double diff_2 = proposal_surface_depths_[surface_id_1][pixel_2] - proposal_surface_depths_[surface_id_2][pixel_2];
// 	  if (diff_1 < 0)
// 	    diff_1 = max(min(diff_1 + penalties_.smoothness_depth_change_threshold, 0.0), -penalties_.smoothness_max_depth_change);
// 	  else
// 	    diff_1 = min(max(diff_1 - penalties_.smoothness_depth_change_threshold, 0.0), penalties_.smoothness_max_depth_change);
// 	  if (diff_2 < 0)
// 	    diff_2 = max(min(diff_2 + penalties_.smoothness_depth_change_threshold, 0.0), -penalties_.smoothness_max_depth_change);
// 	  else
// 	    diff_2 = min(max(diff_2 - penalties_.smoothness_depth_change_threshold, 0.0), penalties_.smoothness_max_depth_change);
        
// 	  //      pairwise_cost += max(diff_1 * diff_2 * penalties_.smoothness_pen_weight, 0.0);
// 	  pairwise_cost += max(sqrt(max(diff_1 * diff_2, 0.0)) * penalties_.smoothness_pen_weight, penalties_.smoothness_small_constant_pen);
//         }
//       }
//     } else if (surface_id_1 < proposal_num_segments_ || surface_id_2 < proposal_num_segments_) {
//       pairwise_cost += penalties_.smoothness_between_empty_pen;
//       for (int other_layer_index = layer_index + 1; other_layer_index < proposal_num_layers_; other_layer_index++) {
// 	int surface_id = -1, other_layer_surface_id = -1, other_layer_other_surface_id = -1;
// 	if (surface_id_1 < proposal_num_segments_) {
// 	  surface_id = surface_id_1;
// 	  other_layer_other_surface_id = layer_labels_2[other_layer_index];
// 	  other_layer_surface_id = layer_labels_1[other_layer_index];
//         } else {
//           surface_id = surface_id_2;
//           other_layer_other_surface_id = layer_labels_1[other_layer_index];
// 	  other_layer_surface_id = layer_labels_2[other_layer_index];
//         }
// 	if (other_layer_other_surface_id == proposal_num_segments_ || other_layer_surface_id < proposal_num_segments_)
// 	  continue;
//         double diff_1 = proposal_surface_depths_[surface_id][pixel_1] - proposal_surface_depths_[other_layer_other_surface_id][pixel_1];
//         double diff_2 = proposal_surface_depths_[surface_id][pixel_2] - proposal_surface_depths_[other_layer_other_surface_id][pixel_2];
// 	if (sqrt(max(diff_1 * diff_2, 0.0) <= penalties_.smoothness_depth_change_threshold))
// 	  pairwise_cost += penalties_.smoothness_break_smooth_connection_pen;
//       }
//     }
//   }
//   //  if (front_layer_cost == penalties_.smoothness_between_empty_pen)
//   //    cout << front_layer_cost + back_layer_cost << endl;
      
//   return pairwise_cost;
// }

void TRWSFusion::solve(const LayerLabelSpace &proposal_label_space, const ParallelFusion::SolutionType<LayerLabelSpace> &current_solution, ParallelFusion::SolutionType<LayerLabelSpace> &solution)
{
  // cout << proposal_surface_depths_[3][35 * 50 + 32] << '\t' << proposal_surface_depths_[3][35 * 50 + 33] << '\t' << proposal_surface_depths_[4][35 * 50 + 32] << '\t' << proposal_surface_depths_[4][35 * 50 + 33] << endl;
  // cout << calcSmoothnessCostMulti(35 * 50 + 32, 35 * 50 + 33, 6 * 49 + 3 * 7 + 5, 6 * 49 + 4 * 7 + 5) << endl;
  // exit(1);
  cout << "fuse" << endl;
  
  proposal_segments_ = proposal_label_space.getSegments();
  proposal_num_segments_ = proposal_segments_.size();
  proposal_num_layers_ = proposal_label_space.getNumLayers();

  if (proposal_num_segments_ == 0) {
    cout << "no segments in fusion" << endl;
    solution = current_solution;
    solution.first = numeric_limits<double>::max();
    return;
  }

  vector<vector<int> > proposal_labels = proposal_label_space.getLabelSpace();
  
  //proposal_surface_depths_ = proposal_surface_depths;
  
  const int NUM_NODES = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_;
  
  unique_ptr<MRFEnergy<TypeGeneral> > energy(new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize()));
  //MRFEnergy<TypeGeneral> *energy = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
  vector<MRFEnergy<TypeGeneral>::NodeId> nodes(NUM_NODES);
  

  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    //cout << pixel << endl;
    vector<int> pixel_proposal = proposal_labels[pixel];
    const int NUM_PROPOSALS = pixel_proposal.size();
    if (NUM_PROPOSALS == 0) {
      cout << "empty proposal error: " << pixel << endl;
      exit(1);
    }
    vector<double> cost(NUM_PROPOSALS);
    for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
      cost[proposal_index] = calcDataCost(pixel, pixel_proposal[proposal_index]);
    nodes[pixel] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&cost[0]));
  }
  
  for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_; i++) {
    vector<int> layer_surface_indicator_proposal = proposal_labels[i];
    const int NUM_PROPOSALS = layer_surface_indicator_proposal.size();
    vector<double> surface_cost(NUM_PROPOSALS);
    for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
      surface_cost[proposal_index] = layer_surface_indicator_proposal[proposal_index] == 1 ? penalties_.surface_pen : 0;
    nodes[i] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&surface_cost[0]));
  }
  
  // for (int i = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_ + proposal_num_layers_; i++) {
  //   vector<int> layer_indicator_proposal = proposal_labels[i];
  //   const int NUM_PROPOSALS = layer_indicator_proposal.size();
  //   vector<double> layer_cost(NUM_PROPOSALS, 0);
  //   for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
  //     layer_cost[proposal_index] = layer_indicator_proposal[proposal_index] == 1 ? penalties_.layer_pen : 0;
  //   nodes[i] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&layer_cost[0]));
  // }

  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposal = proposal_labels[pixel];
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    vector<int> neighbor_pixels;
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;
      vector<int> neighbor_pixel_proposal = proposal_labels[neighbor_pixel];
      vector<double> cost(pixel_proposal.size() * neighbor_pixel_proposal.size(), 0);
      for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++)
	for (int proposal_index_2 = 0; proposal_index_2 < neighbor_pixel_proposal.size(); proposal_index_2++)
          //          cost[label_1 + label_2 * NUM_LABELS_] = calcSmoothnessCost(pixel, neighbor_pixel, label_1, label_2);
          cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = calcSmoothnessCost(pixel, neighbor_pixel, pixel_proposal[proposal_index_1], neighbor_pixel_proposal[proposal_index_2]);
      bool has_non_zero_cost = false;
      for (int i = 0; i < cost.size(); i++)
	if (cost[i] > 0)
	  has_non_zero_cost = true;
      if (has_non_zero_cost == true)
	energy->AddEdge(nodes[pixel], nodes[neighbor_pixel], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));

      // if (cost[0] != cost[1] || cost[1] != cost[2] || cost[2] != cost[3])
      // 	for (int i = 0; i < 4; i++)
      // 	  cout << cost[i] << endl;
    }
  }
  
  bool consider_other_viewpoints = true;
  if (consider_other_viewpoints) {
    map<int, map<int, vector<double> > > pairwise_costs;
    vector<vector<set<int> > > layer_pixel_surface_pixel_pairs = calcOverlapPixels(proposal_labels);
    for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_; layer_index_1++) {
      vector<map<int, vector<int> > > pixel_surface_proposals_map_vec_1(NUM_PIXELS_);
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        vector<int> pixel_proposal = proposal_labels[pixel];
        for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
          int surface_id = *label_it / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index_1)) % (proposal_num_segments_ + 1);
          if (surface_id < proposal_num_segments_)
            pixel_surface_proposals_map_vec_1[pixel][surface_id].push_back(label_it - pixel_proposal.begin());
        }
      }
      vector<set<int> > pixel_surface_pixel_pairs_1 = layer_pixel_surface_pixel_pairs[layer_index_1];
      for (int layer_index_2 = layer_index_1; layer_index_2 < proposal_num_layers_; layer_index_2++) {
        vector<map<int, vector<int> > > pixel_surface_proposals_map_vec_2(NUM_PIXELS_);
	if (layer_index_2 == layer_index_1)
	  pixel_surface_proposals_map_vec_2 = pixel_surface_proposals_map_vec_1;
	else {
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    vector<int> pixel_proposal = proposal_labels[pixel];
	    for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
	      int surface_id = *label_it / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index_2)) % (proposal_num_segments_ + 1);
	      if (surface_id < proposal_num_segments_)
		pixel_surface_proposals_map_vec_2[pixel][surface_id].push_back(label_it - pixel_proposal.begin());
	    }
	  }
	}
        vector<set<int> > pixel_surface_pixel_pairs_2 = layer_pixel_surface_pixel_pairs[layer_index_2];
	for (vector<set<int> >::const_iterator pixel_it = pixel_surface_pixel_pairs_1.begin(); pixel_it != pixel_surface_pixel_pairs_1.end(); pixel_it++) {
	  set<int> surface_pixel_pairs_1 = *pixel_it;
	  set<int> surface_pixel_pairs_2 = pixel_surface_pixel_pairs_2[pixel_it - pixel_surface_pixel_pairs_1.begin()];
	  for (set<int>::const_iterator surface_pixel_pair_it_1 = surface_pixel_pairs_1.begin(); surface_pixel_pair_it_1 != surface_pixel_pairs_1.end(); surface_pixel_pair_it_1++) {
	    for (set<int>::const_iterator surface_pixel_pair_it_2 = surface_pixel_pairs_2.begin(); surface_pixel_pair_it_2 != surface_pixel_pairs_2.end(); surface_pixel_pair_it_2++) {
	      // int surface_id_1 = surface_it_1->first;
	      // int pixel_1 = surface_it_1->second;
	      // int surface_id_2 = surface_it_2->first;
	      // int pixel_2 = surface_it_2->second;
              int surface_id_1 = *surface_pixel_pair_it_1 / NUM_PIXELS_;
              int pixel_1 = *surface_pixel_pair_it_1 % NUM_PIXELS_;
              int surface_id_2 = *surface_pixel_pair_it_2 / NUM_PIXELS_;
              int pixel_2 = *surface_pixel_pair_it_2 % NUM_PIXELS_;
              
	      if (pixel_1 == pixel_2 || surface_id_1 == surface_id_2)
		continue;
	      double cost = 0;
	      if (layer_index_2 == layer_index_1) {
		if (surface_id_2 >= surface_id_1)
		  continue;
		if (abs(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_) <= 1 && abs(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_) <= 1)
		  continue;
		//cout << surface_id_1 << '\t' << surface_id_2 << '\t' << pixel / NUM_PIXELS_ << '\t' << pixel % NUM_PIXELS_ % IMAGE_WIDTH_ << '\t' << pixel % NUM_PIXELS_ / IMAGE_WIDTH_ << endl;
		double depth_diff = abs(proposal_segments_.at(surface_id_1).getDepth(pixel_1) - proposal_segments_.at(surface_id_2).getDepth(pixel_2));
		cost = min(depth_diff * pow(penalties_.smoothness_term_layer_decrease_ratio, proposal_num_layers_ - 1 - layer_index_1) / statistics_.depth_change_smoothness_threshold * penalties_.smoothness_empty_non_empty_ratio, 1.0) * penalties_.other_viewpoint_depth_change_pen + penalties_.smoothness_small_constant_pen;
	      } else {
		if (proposal_segments_.at(surface_id_1).getDepth(pixel_1) > proposal_segments_.at(surface_id_2).getDepth(pixel_2) + statistics_.depth_conflict_threshold)
		  cost = penalties_.other_viewpoint_depth_conflict_pen;
	      }
	      //double cost = (1 - exp(-pow(depth_diff, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2)))) * penalties_.other_viewpoint_depth_change_pen;
	      //double cost = log(2 / (1 + exp(-pow(depth_diff, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.other_viewpoint_depth_change_pen;
	      if (cost < 0.000001)
		continue;
	      // if (pixel_1 >= pixel_2) {
	      //              cout << pixel_1 << '\t' << pixel_2 << '\t' << surface_id_1 << '\t' << surface_id_2 << '\t' << proposal_segments_.at(surface_id_1).getDepth(pixel_1) << '\t' << proposal_segments_.at(surface_id_2).getDepth(pixel_2) << '\t' << layer_index_1 << '\t' << layer_index_2 << endl;
		
              // }

	      if (pixel_1 < pixel_2) {
                if (pairwise_costs.count(pixel_1) == 0 || pairwise_costs[pixel_1].count(pixel_2) == 0)
		  pairwise_costs[pixel_1][pixel_2] = vector<double>(proposal_labels[pixel_1].size() * proposal_labels[pixel_2].size(), 0);
	      } else {
		if (pairwise_costs.count(pixel_2) == 0 || pairwise_costs[pixel_2].count(pixel_1) == 0)
                  pairwise_costs[pixel_2][pixel_1] = vector<double>(proposal_labels[pixel_1].size() * proposal_labels[pixel_2].size(), 0);
	      }
              vector<int> surface_proposals_1 = pixel_surface_proposals_map_vec_1[pixel_1][surface_id_1];
	      vector<int> surface_proposals_2 = pixel_surface_proposals_map_vec_2[pixel_2][surface_id_2];
	      for (vector<int>::const_iterator proposal_it_1 = surface_proposals_1.begin(); proposal_it_1 != surface_proposals_1.end(); proposal_it_1++)
		for (vector<int>::const_iterator proposal_it_2 = surface_proposals_2.begin(); proposal_it_2 != surface_proposals_2.end(); proposal_it_2++)
		  if (pixel_1 < pixel_2)
		    pairwise_costs[pixel_1][pixel_2][*proposal_it_1 + *proposal_it_2 * proposal_labels[pixel_1].size()] += cost;
                  else
		    pairwise_costs[pixel_2][pixel_1][*proposal_it_2 + *proposal_it_1 * proposal_labels[pixel_2].size()] += cost;
	    }
	  }
	}
      }
    }
    
    for (map<int, map<int, vector<double> > >::iterator pixel_it_1 = pairwise_costs.begin(); pixel_it_1 != pairwise_costs.end(); pixel_it_1++)
      for (map<int, vector<double> >::iterator pixel_it_2 = pixel_it_1->second.begin(); pixel_it_2 != pixel_it_1->second.end(); pixel_it_2++)
	energy->AddEdge(nodes[pixel_it_1->first], nodes[pixel_it_2->first], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &pixel_it_2->second[0]));
  }
  
    
  //consider surface cost
  {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      vector<int> pixel_proposal = proposal_labels[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	for (int surface_id = 0; surface_id < proposal_num_segments_; surface_id++) {
	  int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_segments_ + surface_id;
	  
	  vector<int> layer_surface_indicator_proposal = proposal_labels[layer_surface_indicator_index];
	  vector<double> cost(pixel_proposal.size() * layer_surface_indicator_proposal.size(), 0);
	  bool has_non_zero_cost = false;
	  for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++) {
	    for (int proposal_index_2 = 0; proposal_index_2 < layer_surface_indicator_proposal.size(); proposal_index_2++) {
	      int label = pixel_proposal[proposal_index_1];
	      int label_surface_id = label / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_segments_ + 1);
	      double layer_surface_indicator_conflict_cost = (label_surface_id == surface_id && layer_surface_indicator_proposal[proposal_index_2] == 0) ? penalties_.huge_pen : 0;
	      if (layer_surface_indicator_conflict_cost > 0) {
		cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = layer_surface_indicator_conflict_cost;
		has_non_zero_cost = true;
	      }
	    }
	  }

	  if (has_non_zero_cost == true)
	    energy->AddEdge(nodes[pixel], nodes[layer_surface_indicator_index], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
	}
      }
    }
  
    for (int surface_id = 0; surface_id < proposal_num_segments_; surface_id++) {
      for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_; layer_index_1++) {
	int layer_surface_indicator_index_1 = NUM_PIXELS_ + layer_index_1 * proposal_num_segments_ + surface_id;
	vector<int> layer_surface_indicator_proposal_1 = proposal_labels[layer_surface_indicator_index_1];
	for (int layer_index_2 = layer_index_1 + 1; layer_index_2 < proposal_num_layers_; layer_index_2++) {
	  int layer_surface_indicator_index_2 = NUM_PIXELS_ + layer_index_2 * proposal_num_segments_ + surface_id;
	  vector<int> layer_surface_indicator_proposal_2 = proposal_labels[layer_surface_indicator_index_2];  
	  vector<double> cost(layer_surface_indicator_proposal_1.size() * layer_surface_indicator_proposal_2.size(), 0);
	  bool has_non_zero_cost = false;
	  for (int proposal_index_1 = 0; proposal_index_1 < layer_surface_indicator_proposal_1.size(); proposal_index_1++) {
	    for (int proposal_index_2 = 0; proposal_index_2 < layer_surface_indicator_proposal_2.size(); proposal_index_2++) {
	      if (layer_surface_indicator_proposal_1[proposal_index_1] == 1 && layer_surface_indicator_proposal_2[proposal_index_2] == 1)
		cost[proposal_index_1 + proposal_index_2 * layer_surface_indicator_proposal_1.size()] = penalties_.surface_splitted_pen;
	      has_non_zero_cost = true;
	    }
	  }

	  if (has_non_zero_cost == true)
	    energy->AddEdge(nodes[layer_surface_indicator_index_1], nodes[layer_surface_indicator_index_2], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
	}
      }
    }
  }
  
  //consider_layer_cost
  if (false)
  {
    // for (int i = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_ + proposal_num_layers_; i++) {
    //   vector<int> layer_indicator_proposal = proposals[i];
    //   const int NUM_PROPOSALS = layer_indicator_proposal.size();
    //   vector<double> layer_cost(NUM_PROPOSALS, 0);
    //   for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
    //     layer_cost[proposal_index] = layer_indicator_proposal[proposal_index] == 1 ? penalties_.layer_pen : 0;
    //   nodes[i] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&layer_cost[0]));
    // }

    if (true) {
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int layer_indicator_index = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_ + layer_index;
	vector<int> layer_indicator_proposal = proposal_labels[layer_indicator_index];
	for (int surface_id = 0; surface_id < proposal_num_segments_; surface_id++) {
	  int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_segments_ + surface_id;
	  vector<int> layer_surface_indicator_proposal = proposal_labels[layer_surface_indicator_index];
	  vector<double> cost(layer_surface_indicator_proposal.size() * layer_indicator_proposal.size(), 0);
	  bool has_non_zero_cost = false;
	  for (int proposal_index_1 = 0; proposal_index_1 < layer_surface_indicator_proposal.size(); proposal_index_1++) {
	    for (int proposal_index_2 = 0; proposal_index_2 < layer_indicator_proposal.size(); proposal_index_2++) {
	      int label = layer_surface_indicator_proposal[proposal_index_1];
	      int layer_indicator_conflict_cost = (label == 1 && layer_indicator_proposal[proposal_index_2] == 0) ? penalties_.huge_pen : 0;
	      if (layer_indicator_conflict_cost > 0) {
		cost[proposal_index_1 + proposal_index_2 * layer_surface_indicator_proposal.size()] = layer_indicator_conflict_cost;
		has_non_zero_cost = true;
	      }
	    }
	  }
	
	  if (has_non_zero_cost == true)
	    energy->AddEdge(nodes[layer_surface_indicator_index], nodes[layer_indicator_index], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
	}
      }
    } else {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	vector<int> pixel_proposal = proposal_labels[pixel];
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  int layer_indicator_index = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_ + layer_index;
	  vector<int> layer_indicator_proposal = proposal_labels[layer_indicator_index];
	  vector<double> cost(pixel_proposal.size() * layer_indicator_proposal.size(), 0);
	  bool has_non_zero_cost = false;
	  for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++) {
	    for (int proposal_index_2 = 0; proposal_index_2 < layer_indicator_proposal.size(); proposal_index_2++) {
	      int label = pixel_proposal[proposal_index_1];
	      int layer_indicator_conflict_cost = (label / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_segments_ + 1) < proposal_num_segments_ && layer_indicator_proposal[proposal_index_2] == 0) ? penalties_.huge_pen : 0;
	      if (layer_indicator_conflict_cost > 0) {
		cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = layer_indicator_conflict_cost;
		has_non_zero_cost = true;
	      }
	    }
	  }
        
	  if (has_non_zero_cost == true)
	    energy->AddEdge(nodes[pixel], nodes[layer_indicator_index], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
	}
      }
    }
  }

  // if (consider_label_cost == true) {
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     vector<int> pixel_proposal = proposal_labels[pixel];
  //     for (int label = 0; label < pow(proposal_num_segments_ + 1, proposal_num_layers_); label++) {
  //         int label_indicator_index = NUM_PIXELS_ + label;
          
  //         vector<int> label_indicator_proposal = proposal_labels[label_indicator_index];
  //         vector<double> cost(pixel_proposal.size() * label_indicator_proposal.size(), 0);
  //         bool has_non_zero_cost = false;
  //         for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++) {
  //           for (int proposal_index_2 = 0; proposal_index_2 < label_indicator_proposal.size(); proposal_index_2++) {
  //             int pixel_label = pixel_proposal[proposal_index_1];
  //             double label_indicator_conflict_cost = (pixel_label == label && label_indicator_proposal[proposal_index_2] == 0) ? penalties_.label_indicator_conflict_pen : 0;
  //             if (label_indicator_conflict_cost > 0) {
  //               cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = label_indicator_conflict_cost;
  //               has_non_zero_cost = true;
  //             }
  //           }
  //         }
	  
  //         if (has_non_zero_cost == true)
  //           energy->AddEdge(nodes[pixel + pixel_index_offset], nodes[label_indicator_index + indicator_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
  //     }
  //   }
  // }


  const int NUM_INDICATORS = proposal_num_layers_ * proposal_num_segments_; // + proposal_num_layers_;
  vector<int> fixed_indicator_mask(NUM_INDICATORS, -1);
  int num_fixed_indicators = 0;
  map<int, set<int> > surface_layers;

  if (consider_surface_cost_)
  {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      vector<int> pixel_proposal = proposal_labels[pixel];
      for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++) {
	int label = pixel_proposal[proposal_index];
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  int surface_id = label / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_segments_ + 1);
	  if (surface_id < proposal_num_segments_) {
	    // if (surface_layers[surface_id].count(layer_index) == 0)
	    //   cout << layer_index << '\t' << surface_id << endl;
	    surface_layers[surface_id].insert(layer_index);
	  }
	}
      }
    }
  
    for (map<int, set<int> >::const_iterator surface_it = surface_layers.begin(); surface_it != surface_layers.end(); surface_it++) {
      set<int> layers = surface_it->second;
      if (layers.size() == proposal_num_segments_)
	continue;
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	if (layers.count(layer_index) > 0)
	  continue;
	int indicator_index = layer_index * proposal_num_segments_ + surface_it->first;
	vector<double> fixed_indicator_cost_diff(2, 0);
	fixed_indicator_cost_diff[1] = 1000000;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
	fixed_indicator_mask[indicator_index] = 0;
	num_fixed_indicators++;
      }
      // if (layers.size() == 1) {
      //   int layer_index = *layers.begin();
      //   int indicator_index = layer_index * proposal_num_segments_ + surface_it->first;
      //   vector<double> fixed_indicator_cost_diff(2, 0);
      //   fixed_indicator_cost_diff[0] = 1000000;
      //   energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
      //   fixed_indicator_mask[indicator_index] = 1;
      //   num_fixed_indicators++;
      // }  
    }
  }

  
  
  MRFEnergy<TypeGeneral>::Options options;
  options.m_iterMax = 2000;
  options.m_printIter = 200;
  options.m_printMinIter = 100;
  options.m_eps = 0.1;

  //energy->SetAutomaticOrdering();
  //energy->ZeroMessages();
  //energy->AddRandomMessages(0, 0, 0.001);
  
  //double lower_bound;
  
  // static int proposal_index = 0;
  // if (proposal_index == 200) {
  //   solution_.assign(NUM_NODES, 0);
  //   for (int i = 0; i < NUM_NODES; i++)
  //     solution_[i] = proposal_labels[i][0];
  //   checkSolutionEnergy(solution_);
  //   return solution_;
  // }
  // proposal_index++;

  double lower_bound = -1, energy_value = -1;
  energy->Minimize_TRW_S(options, lower_bound, energy_value);
  
  vector<int> fused_labels(NUM_NODES);
  vector<double> confidences(NUM_NODES);
  //vector<double> solution_labels(NUM_NODES);
  for (int i = 0; i < NUM_NODES; i++) {
    // if (consider_surface_cost == false && i >= NUM_PIXELS_)
    //   break;
    // if (consider_layer_cost == false && i >= NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_)
    //   break;
    // if (consider_surface_cost == false && consider_layer_cost == false && i >= NUM_PIXELS_)
    //   break;
    int label = i < NUM_PIXELS_ ? energy->GetSolution(nodes[i]) : energy->GetSolution(nodes[i]);
    //double confidence = i < NUM_PIXELS_ ? energy->GetConfidence(i + pixel_index_offset) : energy->GetConfidence(i + indicator_index_offset);
    //solution_labels[i] = label;
    fused_labels[i] = proposal_labels[i][label];
    //confidences[i] = confidence;

    
    // if (i >= NUM_PIXELS_ && i - NUM_PIXELS_ < proposal_num_layers_ * proposal_num_segments_) {
    // //   //if (label == 1)
    //   cout << "layer: " << (i - NUM_PIXELS_) / proposal_num_segments_ << "\tsurface: " << (i - NUM_PIXELS_) % proposal_num_segments_ << '\t' << fused_labels[i] << '\t' << confidence << endl;
    // }

    
    //cout << confidences[i] << endl;
    //      cout << solution_[i] << endl;
  }
  
  //cout << "energy: " << energy_ << " lower bound: " << lower_bound << endl;

  
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   fused_labels[pixel] = proposal_labels[pixel][1];

  
  checkSolutionEnergy(fused_labels);
  
  //  exit(1);
  // vector<int> test_labels = fused_labels;
  // //test_labels[NUM_PIXELS_ + 26] = 1;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   if (test_labels[pixel] / (proposal_num_segments_ + 1) / (proposal_num_segments_ + 1) % (proposal_num_segments_ + 1) == 4)
  //     test_labels[pixel] = 4 * pow(proposal_num_segments_ + 1, 1) + proposal_num_segments_ * pow(proposal_num_segments_ + 1, 2) + test_labels[pixel] % (proposal_num_segments_ + 1);
      //test_labels[pixel] += -2 * (proposal_num_segments_ + 1) + 2 * pow(proposal_num_segments_ + 1, 2);
  //test_labels[NUM_PIXELS_ + 36] = 1;
  //test_labels[NUM_PIXELS_ + 37] = 1;
  //  test_labels = getOptimalSolution();
  //  test_labels[29 * 50 + 32] += (4 - 9) * 100 + (2 - 4) * 10;
  //checkSolutionEnergy(test_labels);
  
  const double OPTIMAL_THRESHOLD_SCALE = 1.2;
  const double LOWER_BOUND_DIFF_THRESHOLD = 0.01;

  if (energy_value <= lower_bound * OPTIMAL_THRESHOLD_SCALE && energy_value < current_solution.first) {
    LayerLabelSpace solution_label_space = proposal_label_space;
    solution_label_space.setSingleLabels(fused_labels);
    solution = make_pair(energy_value, solution_label_space);
  } else {
    cout << "Too large energy: " << energy_value << '\t' << lower_bound << endl;
    solution = current_solution;
    //    solution.setLabelSpace(current_solution);
  }
}

double TRWSFusion::checkSolutionEnergy(const vector<int> &solution_for_check)
{
  vector<int> solution = solution_for_check;

  if (consider_surface_cost_) {
    vector<int> correct_indicators(proposal_num_segments_ * proposal_num_layers_, 0);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int label = solution[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int surface_id = label / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_segments_ + 1);
	if (surface_id < proposal_num_segments_) {
	  correct_indicators[proposal_num_segments_ * layer_index + surface_id] = 1;
	}
      }
    }
    bool has_indicator_conflict = false;
    for (int indicator_index = 0; indicator_index < proposal_num_segments_ * proposal_num_layers_; indicator_index++) {
      if (solution[indicator_index + NUM_PIXELS_] != correct_indicators[indicator_index]) {
	has_indicator_conflict = true;
	//cout << "correct indicator: " << indicator_index << '\t' << proposal_num_segments_ << '\t' << solution[indicator_index + NUM_PIXELS_] << endl;
	solution[indicator_index + NUM_PIXELS_] = correct_indicators[indicator_index];
	//break;
      }
    }
    //assert(has_indicator_conflict == false);
  }
  
  
  // for (int segment_id = 0; segment_id < 4; segment_id++) {
  //   for (int pixel = 8; pixel < 12; pixel++)
  //     cout << proposal_segments_[segment_id].getDepth(pixel) << '\t';
  //   cout << endl;
  // }
  // cout << exp(-pow(calcColorDiff(8, 9), 2) / (2 * color_diff_var_)) * penalties_.smoothness_boundary_pen << '\t' << calcSmoothnessCost(8, 9, solution[8], solution[11]) << '\t' << calcSmoothnessCost(8, 9, solution[8], solution[9]) << endl;
  // cout << exp(-pow(calcColorDiff(9, 10), 2) / (2 * color_diff_var_)) * penalties_.smoothness_boundary_pen << '\t' << calcSmoothnessCost(9, 10, solution[8], solution[11]) << '\t' << calcSmoothnessCost(9, 10, solution[9], solution[10]) << endl;
  // cout << exp(-pow(calcColorDiff(10, 11), 2) / (2 * color_diff_var_)) * penalties_.smoothness_boundary_pen << '\t' << calcSmoothnessCost(10, 11, solution[8], solution[11]) << '\t' << calcSmoothnessCost(10, 11, solution[10], solution[11]) << endl;
  // exit(1);

  bool test_energy = false;
  if (test_energy) {
    static int checking_index = 0;
    stringstream energy_filename;
    energy_filename << "Test/energy_" << checking_index;
    ofstream energy_out_str(energy_filename.str().c_str());
    checking_index++;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      energy_out_str << calcDataCost(pixel, solution[pixel]) << endl;;

    // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    //   int x = pixel % IMAGE_WIDTH_;
    //   int y = pixel / IMAGE_WIDTH_;
    //   vector<int> neighbor_pixels;
    //   if (x < IMAGE_WIDTH_ - 1)
    // 	neighbor_pixels.push_back(pixel + 1);
    //   if (y < IMAGE_HEIGHT_ - 1)
    // 	neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    //   if (x > 0 && y < IMAGE_HEIGHT_ - 1)
    // 	neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    //   if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
    // 	neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);

    //   double cost_sum = 0;
    //   for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
    // 	int neighbor_pixel = *neighbor_pixel_it;
    // 	//    	if ((solution[pixel] / ((int)pow(proposal_num_segments_ + 1, 0)) % (proposal_num_segments_ + 1) == 0 && solution[neighbor_pixel] / ((int)pow(proposal_num_segments_ + 1, 0)) % (proposal_num_segments_ + 1) == 2)
    // 	//|| (solution[pixel] / ((int)pow(proposal_num_segments_ + 1, 0)) % (proposal_num_segments_ + 1) == 2 && solution[neighbor_pixel] / ((int)pow(proposal_num_segments_ + 1, 0)) % (proposal_num_segments_ + 1) == 0)) {
    // 	//double cost_1 = log(2 / (1 + boundary_scores_[pixel] * exp(-1 / (2 * pow(penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
    // 	//    	  double cost_2 = log(2 / (1 + boundary_scores_[neighbor_pixel] * exp(-1 / (2 * pow(penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
    // 	  //energy_out_str << pixel << '\t' << calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]) << endl;
    // 	  cost_sum += calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]);
    //       //energy_out_str << pixel << '\t' << neighbor_pixel << '\t' << calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]) << endl;
    // 	  //        }
    //   }
    //   energy_out_str << cost_sum << endl;
    // }
    energy_out_str.close();
  }
  
  
  // if (checking_index == 2) {
  //   ifstream energy_in_str_1("Test/energy_1");
  //   ifstream energy_in_str_2("Test/energy_2");
  //   ofstream energy_diff_out_str("Test/energy_diff");
  //   cout << "yes" << endl;
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     double energy_1;
  //     energy_in_str_1 >> energy_1;
  //     double energy_2;
  //     energy_in_str_2 >> energy_2;
  //     energy_diff_out_str << pixel << '\t' << energy_1 - energy_2 << endl;
  //   }
  //   energy_diff_out_str.close();  
  //   exit(1);
  // }

  double unary_cost = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    unary_cost += calcDataCost(pixel, solution[pixel]);
  
  double pairwise_cost = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    vector<int> neighbor_pixels;
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;
      pairwise_cost += calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]);
      
      //energy_out_str << pixel << '\t' << neighbor_pixel << '\t' << calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]) << endl;
      
      // if (solution[pixel] / (int)pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1) == 3 && solution[neighbor_pixel] / (int)pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1) == 4)
      // 	cout << calcSmoothnessCostMulti(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]) << endl;
    }
  }

  double other_viewpoint_depth_change_cost = 0;
  bool consider_other_viewpoints = true;
  if (consider_other_viewpoints) {

    vector<vector<int> > solution_labels(solution.size());
    for (int i = 0; i < solution.size(); i++)
      solution_labels[i].push_back(solution[i]);
    vector<vector<set<int> > > layer_pixel_surface_pixel_pairs = calcOverlapPixels(solution_labels);

    for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_; layer_index_1++) {
      vector<set<int> > pixel_surface_pixel_pairs_1 = layer_pixel_surface_pixel_pairs[layer_index_1];
      for (int layer_index_2 = layer_index_1; layer_index_2 < proposal_num_layers_; layer_index_2++) {
        vector<set<int> > pixel_surface_pixel_pairs_2 = layer_pixel_surface_pixel_pairs[layer_index_2];
        for (vector<set<int> >::const_iterator pixel_it = pixel_surface_pixel_pairs_1.begin(); pixel_it != pixel_surface_pixel_pairs_1.end(); pixel_it++) {
          set<int> surface_pixel_pairs_1 = *pixel_it;
          set<int> surface_pixel_pairs_2 = pixel_surface_pixel_pairs_2[pixel_it - pixel_surface_pixel_pairs_1.begin()];
          for (set<int>::const_iterator surface_pixel_pair_it_1 = surface_pixel_pairs_1.begin(); surface_pixel_pair_it_1 != surface_pixel_pairs_1.end(); surface_pixel_pair_it_1++) {
            for (set<int>::const_iterator surface_pixel_pair_it_2 = surface_pixel_pairs_2.begin(); surface_pixel_pair_it_2 != surface_pixel_pairs_2.end(); surface_pixel_pair_it_2++) {
              // int surface_id_1 = surface_it_1->first;
              // int pixel_1 = surface_it_1->second;
              // int surface_id_2 = surface_it_2->first;
              // int pixel_2 = surface_it_2->second;
              int surface_id_1 = *surface_pixel_pair_it_1 / NUM_PIXELS_;
              int pixel_1 = *surface_pixel_pair_it_1 % NUM_PIXELS_;
              int surface_id_2 = *surface_pixel_pair_it_2 / NUM_PIXELS_;
              int pixel_2 = *surface_pixel_pair_it_2 % NUM_PIXELS_;

	      if (pixel_1 == pixel_2 || surface_id_1 == surface_id_2)
                continue;
              double cost = 0;
              if (layer_index_2 == layer_index_1) {
                if (surface_id_1 >= surface_id_2)
                  continue;
                if (abs(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_) <= 1 && abs(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_) <= 1)
                  continue;
                double depth_diff = abs(proposal_segments_.at(surface_id_1).getDepth(pixel_1) - proposal_segments_.at(surface_id_2).getDepth(pixel_2));
                cost = min(depth_diff / statistics_.depth_change_smoothness_threshold * penalties_.smoothness_empty_non_empty_ratio, 1.0) * penalties_.other_viewpoint_depth_change_pen + penalties_.smoothness_small_constant_pen;
              } else {
                if (proposal_segments_.at(surface_id_1).getDepth(pixel_1) > proposal_segments_.at(surface_id_2).getDepth(pixel_2) + statistics_.depth_conflict_threshold) {
                  cost = penalties_.other_viewpoint_depth_conflict_pen;
		  cout << "other viewpoint cost: " << pixel_1 << '\t' << pixel_2 << '\t' << proposal_segments_.at(surface_id_1).getDepth(pixel_1) << '\t' << proposal_segments_.at(surface_id_2).getDepth(pixel_2) << endl;
		}
              }
	      other_viewpoint_depth_change_cost += cost;
	    }
	  }
        }
      }
    }
  }

  double surface_cost = 0;
  double layer_cost = 0;
  if (consider_surface_cost_) {
    if (false) {
      for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + proposal_num_segments_; i++) {
        int segment_layer = solution[i];
	surface_cost += calcNumOneBits(segment_layer) * penalties_.surface_pen;
      }
    } else {
      for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_; i++) {
	int layer_surface_indicator = solution[i];
	surface_cost += layer_surface_indicator == 1 ? penalties_.surface_pen : 0;
      }
    
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	int pixel_label = solution[pixel];
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  for (int surface_id = 0; surface_id < proposal_num_segments_; surface_id++) {
	    int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_segments_ + surface_id;
	  
	    int layer_surface_indicator = solution[layer_surface_indicator_index];
	    int label_surface_id = pixel_label / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_segments_ + 1);
	    surface_cost += (label_surface_id == surface_id && layer_surface_indicator == 0) ? penalties_.huge_pen : 0;
	  }
	}
      }

      for (int surface_id = 0; surface_id < proposal_num_segments_; surface_id++) {
	for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_; layer_index_1++) {
	  int layer_surface_indicator_index_1 = NUM_PIXELS_ + layer_index_1 * proposal_num_segments_ + surface_id;
	  int layer_surface_indicator_1 = solution[layer_surface_indicator_index_1];
	  for (int layer_index_2 = layer_index_1 + 1; layer_index_2 < proposal_num_layers_; layer_index_2++) {
	    int layer_surface_indicator_index_2 = NUM_PIXELS_ + layer_index_2 * proposal_num_segments_ + surface_id;
	    int layer_surface_indicator_2 = solution[layer_surface_indicator_index_2];
	    surface_cost += layer_surface_indicator_1 == 1 && layer_surface_indicator_2 == 1 ? penalties_.surface_splitted_pen : 0;
	  }
	}
      }

    
      // for (int i = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_ + proposal_num_layers_; i++) {
      // 	int layer_indicator = solution[i];
      // 	layer_cost += layer_indicator == 1 ? penalties_.layer_pen : 0;
      // }

      // for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
      // 	int layer_indicator_index = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_segments_ + layer_index;
      // 	int layer_indicator = solution[layer_indicator_index];
      // 	for (int surface_id = 0; surface_id < proposal_num_segments_; surface_id++) {
      // 	  int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_segments_ + surface_id;
      // 	  int layer_surface_indicator = solution[layer_surface_indicator_index];
      // 	  layer_cost += (layer_surface_indicator == 1 && layer_indicator == 0) ? penalties_.huge_pen : 0;

      // 	  // if (layer_surface_indicator == 1 && layer_indicator == 0)
      // 	  //   cout << layer_index << '\t' << surface_id << endl;
      // 	}
      // }
    }
  }
  
  double total_cost = unary_cost + pairwise_cost + other_viewpoint_depth_change_cost + surface_cost;
  cout << "cost: " << total_cost << " = " << unary_cost << " + " << pairwise_cost << " + " << other_viewpoint_depth_change_cost << " + " << surface_cost << endl;
  return total_cost;
}

vector<int> TRWSFusion::getOptimalSolution()
{
  set<int> background_surfaces;
  background_surfaces.insert(0);
  background_surfaces.insert(1);
  background_surfaces.insert(2);
  background_surfaces.insert(6);
  vector<int> background_layer(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    double min_depth = 1000000;
    int min_depth_surface = -1;
    for (set<int>::const_iterator surface_it = background_surfaces.begin(); surface_it != background_surfaces.end(); surface_it++) {
      double depth = proposal_segments_[*surface_it].getDepth(pixel);
      if (depth > 0 && depth < min_depth) {
	min_depth = depth;
	min_depth_surface = *surface_it;
      }
    }
    background_layer[pixel] = min_depth_surface;
  }

  vector<int> labels(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int segment_id = -1; //segmentation_[pixel];
    if (segment_id == 3)
      labels[pixel] = segment_id * (proposal_num_segments_ + 1) * (proposal_num_segments_ + 1) + proposal_num_segments_ * (proposal_num_segments_ + 1) + background_layer[pixel];
    else if (segment_id == 4 || segment_id == 5)
      labels[pixel] = proposal_num_segments_ * (proposal_num_segments_ + 1) * (proposal_num_segments_ + 1) + segment_id * (proposal_num_segments_ + 1) + background_layer[pixel];
    else
      labels[pixel] = proposal_num_segments_ * (proposal_num_segments_ + 1) * (proposal_num_segments_ + 1) + proposal_num_segments_ * (proposal_num_segments_ + 1) + background_layer[pixel];
  }
  return labels;
}

// void TRWSFusion::calcDepthSVar()
// {
//   vector<double> depths(NUM_PIXELS_);
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
//     depths[pixel] = point_cloud_[pixel * 3 + 2];
  
//   double mean;
//   calcStatistics(depths, mean, depth_svar_);
// }

void TRWSFusion::calcBoundaryScores()
{
  static bool boundary_score_calculated = false;
  if (boundary_score_calculated == true)
    return;
  boundary_score_calculated = true;
  boundary_scores_ = vector<double>(NUM_PIXELS_, 1);
  return;
  
  Mat blurred_image;
  GaussianBlur(image_, blurred_image, cv::Size(3, 3), 0, 0);
  Mat gray_image;
  cvtColor(blurred_image, gray_image, CV_RGB2GRAY);
  Mat image_gradient_x, image_gradient_y, image_gradient_x_abs, image_gradient_y_abs;
  Sobel(gray_image, image_gradient_x, CV_16S, 1, 0, 3);
  Sobel(gray_image, image_gradient_y, CV_16S, 0, 1, 3);
  convertScaleAbs(image_gradient_x, image_gradient_x_abs);
  convertScaleAbs(image_gradient_y, image_gradient_y_abs);

  vector<double> gradients(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    gradients[pixel] = sqrt(pow(image_gradient_x_abs.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_), 2) + pow(image_gradient_y_abs.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_), 2));
  
  double gradient_mean, gradient_svar;
  calcStatistics(gradients, gradient_mean, gradient_svar);
  // double minimum_score = 0.5;
  // double maximum_score = 1;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   boundary_scores_[pixel] = (minimum_score + maximum_score) / 2 + max(min((boundary_scores_[pixel] - boundary_score_mean) / boundary_score_svar, 1.0), -1.0) * (maximum_score - minimum_score) / 2;

  boundary_scores_.assign(NUM_PIXELS_, 0);
  double boundary_score_gradient_weight = 0.2;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    boundary_scores_[pixel] = (1 - boundary_score_gradient_weight) + boundary_score_gradient_weight * normalizeStatistically(gradients[pixel], gradient_mean, gradient_svar, 0.3, 0.3);
  }
  
  //boundary_scores_[pixel] = 1 - gradient_weight * exp(-pow(boundary_scores_[pixel] / (boundary_score_svar * 2), 2) / 2);
  
  // double intensity_mean, intensity_svar;
  // vector<double> intensities(NUM_PIXELS_);
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   intensities[pixel] = gray_image.at<uchar>(pixel);
  // calcStatistics(intensities, intensity_mean, intensity_svar);
  
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   boundary_scores_[pixel] = max(min(boundary_scores_[pixel] / intensity_svar, 1.0), 0.1);

  Mat boundary_score_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  for (int y = 0; y < IMAGE_HEIGHT_; y++)
    for (int x = 0; x < IMAGE_WIDTH_; x++)
      boundary_score_image.at<uchar>(y, x) = boundary_scores_[y * IMAGE_WIDTH_ + x] * 255;
  imwrite("Test/boundary_score_image.bmp", boundary_score_image);
}

// void TRWSFusion::calcPixelConfidences()
// {
//   static bool pixel_confidence_calculated = false;
//   if (pixel_confidence_calculated == true)
//     return;
// }

void TRWSFusion::calcColorDiffVar()
{
  Mat blurred_image;
  GaussianBlur(image_, blurred_image, cv::Size(3, 3), 0, 0);
  //blurred_hsv_image_ = blurred_image.clone();
  blurred_image.convertTo(blurred_hsv_image_, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image_, blurred_hsv_image_, CV_BGR2HSV);

  // imshow("blurred hsv image", blurred_hsv_image_);
  // waitKey();
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   Vec3f color = blurred_hsv_image_.at<Vec3f>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
  //   if (color[0] == 0 && color[1] == 0)
  //     cout << pixel << '\t' << color << endl;
  // }
  // exit(1);
  
  double color_diff_sum2 = 0;
  double depth_diff_sum2 = 0;
  int num_pairs = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    double depth = point_cloud_[pixel * 3 + 2];
    if (depth < 0)
      continue;
    vector<int> neighbor_pixels;
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;  
      double neighbor_depth = point_cloud_[neighbor_pixel * 3 + 2];
      if (neighbor_depth < 0)
	continue;
      color_diff_sum2 += pow(calcColorDiff(pixel, neighbor_pixel), 2);
      depth_diff_sum2 += pow(neighbor_depth - depth, 2);
      num_pairs++;
    }
  }
  color_diff_var_ = color_diff_sum2 / num_pairs;
  cout << "color diff var: " << color_diff_var_ << endl;
  cout << "depth diff var: " << depth_diff_sum2 / num_pairs << endl;
  //depth_diff_var_ = depth_diff_sum2 / num_pairs;
}

double TRWSFusion::calcColorDiff(const int pixel_1, const int pixel_2)
{
  Vec3f color_1 = blurred_hsv_image_.at<Vec3f>(pixel_1 / IMAGE_WIDTH_, pixel_1 % IMAGE_WIDTH_);
  Vec3f color_2 = blurred_hsv_image_.at<Vec3f>(pixel_2 / IMAGE_WIDTH_, pixel_2 % IMAGE_WIDTH_);

  double color_diff = sqrt(pow(color_1[1] * cos(color_1[0] * M_PI / 180) - color_2[1] * cos(color_2[0] * M_PI / 180), 2) + pow(color_1[1] * sin(color_1[0] / 180 * M_PI) - color_2[1] * sin(color_2[0] / 180 * M_PI), 2)); // + pow(color_1[2] * 0.1 - color_2[2] * 0.1, 2));
			   
  // Vec3b color_1 = blurred_image_.at<Vec3b>(pixel_1 / IMAGE_WIDTH_, pixel_1 % IMAGE_WIDTH_);
  // Vec3b color_2 = blurred_image_.at<Vec3b>(pixel_2 / IMAGE_WIDTH_, pixel_2 % IMAGE_WIDTH_);
  // double color_diff = 0;
  // for (int c = 0; c < 3; c++)
  //   color_diff += pow(color_1[c] - color_2[c], 2);
  // color_diff = sqrt(color_diff / 3);
  return color_diff;
}

vector<vector<set<int> > > TRWSFusion::calcOverlapPixels(const vector<vector<int> > &proposal_labels)
{
  // for (map<int, Segment>::const_iterator segment_it = proposal_segments_.begin(); segment_it != proposal_segments_.end(); segment_it++)
  //   cout << segment_it->first << endl;
  
  vector<vector<set<int> > > layer_pixel_surface_pixel_pairs(proposal_num_layers_, vector<set<int> >(NUM_PIXELS_ * 4));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposal = proposal_labels[pixel];
    for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int surface_id = *label_it / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_segments_ + 1);
        if (surface_id == proposal_num_segments_)
	  continue;
	
	// if (layer_index != 2)
	//   cout << *label_it << endl;
        vector<int> projected_pixels = proposal_segments_.at(surface_id).projectToOtherViewpoints(pixel, statistics_.viewpoint_movement);
        for (vector<int>::const_iterator projected_pixel_it = projected_pixels.begin(); projected_pixel_it != projected_pixels.end(); projected_pixel_it++) {
          // if (layer_pixel_surface_pixel_maps[layer_index][*projected_pixel_it].count(surface_id) == 0)
	  //   layer_pixel_surface_pixels_maps[layer_index][*projected_pixel_it][surface_id] = pixel;
          // if (proposal_segments_.at(surface_id).getDepth(pixel) < proposal_segments_.at(surface_id).getDepth(layer_pixel_surface_pixels_maps[layer_index][*projected_pixel_it][surface_id]))
          //   cout << pixel << '\t' << surface_id << endl;
	  layer_pixel_surface_pixel_pairs[layer_index][*projected_pixel_it].insert(surface_id * NUM_PIXELS_ + pixel);
        }
      }
    }
  }
  
  // bool check_projected_image = false;
  // if (check_projected_image) {
  //   Mat projected_pixel_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     vector<int> pixel_proposal = proposal_labels[pixel];
  //     for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
  // 	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
  // 	  int surface_id = *label_it / static_cast<int>(pow(proposal_num_segments_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_segments_ + 1);
  // 	  if (surface_id != 10)
  // 	    continue;
  // 	  vector<int> projected_pixels = proposal_segments_.at(surface_id).projectToOtherViewpoints(pixel, statistics_.viewpoint_movement);
  // 	  for (vector<int>::const_iterator projected_pixel_it = projected_pixels.begin(); projected_pixel_it != projected_pixels.end(); projected_pixel_it++) {
  // 	    int projected_pixel = *projected_pixel_it % NUM_PIXELS_;
  // 	    int direction = *projected_pixel_it / NUM_PIXELS_;
  // 	    if (direction != 1)
  // 	      continue;
  // 	    Vec3b color((direction % 3 == 0) * 255, (direction % 3 == 1 || direction == 3) * 255, (direction % 3 == 2) * 255);
  // 	    Vec3b previous_color = projected_pixel_image.at<Vec3b>(projected_pixel / IMAGE_WIDTH_, projected_pixel % IMAGE_WIDTH_);
  // 	    if (previous_color != Vec3b(0, 0, 0) && previous_color != color)
  // 	      color = Vec3b(255, 255, 255);
  // 	    projected_pixel_image.at<Vec3b>(projected_pixel / IMAGE_WIDTH_, projected_pixel % IMAGE_WIDTH_) = color;
  // 	  }
  // 	}
  //     }
  //   }
  //   imwrite("Test/projected_pixel_image_2.bmp", projected_pixel_image);
  //   exit(1);
  // }
  return layer_pixel_surface_pixel_pairs;
}
