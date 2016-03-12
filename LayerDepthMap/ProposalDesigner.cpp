#include "ProposalDesigner.h"

#include <iostream>

#include "utils.h"
#include "../base/cv_utils/cv_utils.h"
#include "ConcaveHullFinder.h"
#include "StructureFinder.h"
#include <glog/logging.h>
//#include "LayerEstimator.h"

//#include "PointCloudSegmenter.h"


using namespace std;
using namespace cv;
using namespace cv_utils;

ProposalDesigner::ProposalDesigner(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const vector<double> &camera_parameters, const int num_layers, const RepresenterPenalties penalties, const DataStatistics statistics, const int scene_index, const bool use_concave_hull_proposal_first) : image_(image), point_cloud_(point_cloud), normals_(normals), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), CAMERA_PARAMETERS_(camera_parameters), penalties_(penalties), statistics_(statistics), NUM_PIXELS_(image.cols * image.rows), NUM_LAYERS_(num_layers), SCENE_INDEX_(scene_index), NUM_ALL_PROPOSAL_ITERATIONS_(3), proposal_iteration_(0), use_concave_hull_proposal_first_(use_concave_hull_proposal_first)
{
  binary_proposal_designer_ = unique_ptr<BinaryProposalDesigner>(new BinaryProposalDesigner(image, point_cloud, normals, camera_parameters, num_layers, penalties, statistics, scene_index));
  //layer_inpainter_ = unique_ptr<LayerInpainter>(new LayerInpainter(image_, segmentation_, surface_depths_, penalties_, false, true));
  //layer_estimator_ = unique_ptr<LayerEstimator>(new LayerEstimator(image_, point_cloud_, segmentation_, surface_depths_, NUM_LAYERS_, penalties_, surface_colors_, SCENE_INDEX_));
  //  segment_graph_ = layer_estimator_->getSegmentGraph();
  
  //  calcSegmentations();

  Mat blurred_image;
  GaussianBlur(image_, blurred_image, cv::Size(3, 3), 0, 0);
  blurred_image.convertTo(blurred_hsv_image_, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image_, blurred_hsv_image_, CV_BGR2HSV);

  initializeCurrentSolution();

  proposal_type_indices_ = vector<int>(7);
  for (int c = 0; c < 7; c++)
    proposal_type_indices_[c] = c;
  proposal_type_index_ptr_ = -1;
  all_proposal_iteration_ = 0;
}

ProposalDesigner::~ProposalDesigner()
{
}

void ProposalDesigner::setCurrentSolution(const LayerLabelSpace &current_solution_label_space)
{
  vector<int> current_solution_labels(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    current_solution_labels[pixel] = current_solution_label_space.getLabelOfNode(pixel)[0];
  int current_solution_num_surfaces = current_solution_label_space.getNumSegments();
  map<int, Segment> current_solution_segments = current_solution_label_space.getSegments();

  if (current_solution_num_surfaces == 0) {
    current_solution_labels_ = current_solution_labels;
    current_solution_num_surfaces_ = current_solution_num_surfaces;
    current_solution_segments_ = current_solution_segments;
    return;
  }

  //cout << "set current solution" << endl;
  
  //current_solution_ = current_solution;
  //current_solution_num_surfaces_ = current_solution_num_surfaces;
  //current_solution_surface_depths_ = current_solution_surface_depths;

  map<int, int> surface_id_map;
  int new_surface_id = 0;
for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      if (surface_id == current_solution_num_surfaces)
        continue;
      if (surface_id_map.count(surface_id) == 0) {
        surface_id_map[surface_id] = new_surface_id;
        new_surface_id++;
      }
    }
    if (surface_id_map.size() == current_solution_num_surfaces)
      break;
  }
  surface_id_map[current_solution_num_surfaces] = new_surface_id;


  current_solution_segments_.clear();
  for (map<int, Segment>::const_iterator segment_it = current_solution_segments.begin(); segment_it != current_solution_segments.end(); segment_it++) {
    if (surface_id_map.count(segment_it->first) > 0) {
      current_solution_segments_[surface_id_map[segment_it->first]] = segment_it->second;
      //current_solution_segments_[surface_id_map[segment_it->first]].setVisiblePixels(segment_visible_pixels[surface_id_map[segment_it->first]]);
    }
  }

  
  vector<int> new_current_solution_labels(NUM_PIXELS_);
  int new_current_solution_num_surfaces = new_surface_id;
  
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    if (checkLabelValidity(pixel, current_solution_label, current_solution_num_surfaces, current_solution_segments) == false) {
      cout << "invalid current label at pixel: " << pixel << endl;
      exit(1);
    }
    int new_label = 0;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      // if (pixel == 0)
      //   cout << surface_id << endl;
      new_label += surface_id_map[surface_id] * pow(new_current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index);
    }
    new_current_solution_labels[pixel] = new_label;
  }

  // vector<int> segment_layer_labels(new_current_solution_num_surfaces);
  // for (map<int, int>::const_iterator segment_it = surface_id_map.begin(); segment_it != surface_id_map.end(); segment_it++)
  //   segment_layer_labels[segment_it->second] = current_solution_labels[NUM_PIXELS_ + segment_it->first];
  // new_current_solution_labels.insert(new_current_solution_labels.end(), segment_layer_labels.begin(), segment_layer_labels.end());
  
  
  current_solution_labels_ = new_current_solution_labels;
  current_solution_num_surfaces_ = new_current_solution_num_surfaces;
  
  // writeDispImageFromSegments(current_solution_labels_, current_solution_num_surfaces_, current_solution_segments_, NUM_LAYERS_, IMAGE_WIDTH_, IMAGE_HEIGHT_, "Test/disp_image_0_new.bmp");
  // Mat disp_image_1 = imread("Test/disp_image_0.bmp", 0);
  // Mat disp_image_2 = imread("Test/disp_image_0_new.bmp", 0);
  // for (int y = 0; y < IMAGE_HEIGHT_; y++) {
  //   for (int x = 0; x < IMAGE_WIDTH_; x++) {
  //     int disp_1 = disp_image_1.at<uchar>(y, x);
  //     int disp_2 = disp_image_2.at<uchar>(y, x);
  //     if (disp_1 != disp_2)
  // 	cout << disp_1 - disp_2 << endl;
  //   }
  // }
  // exit(1);
}

void ProposalDesigner::getProposals(LayerLabelSpace &proposal_label_space, const LayerLabelSpace &current_solution, const int N)
{
  setCurrentSolution(current_solution);
  
  // if (iteration != 2) {
  //   cout << current_solution_num_surfaces_ << '\t' << current_solution_segments_.size() << endl;
  //   exit(1);
  // }
  // bool test = false;
  // if (test) {
  //   //generateSegmentRefittingProposal();
  //   generateSingleSurfaceExpansionProposal(8);
  //   //generateSingleProposal();
  //   proposal_labels = proposal_labels_;
  //   proposal_num_surfaces = proposal_num_surfaces_;
  //   proposal_segments = proposal_segments_;
  //   proposal_type = proposal_type_;
  //   return true;
  // }
  
  const int NUM_PROPOSAL_TYPES = 7;
  
  const bool RANDOM_PROPOSAL = false;
  if (RANDOM_PROPOSAL) {
    for (int label_space_index = 0; label_space_index < N; label_space_index++) {
      bool generate_success = false;
      if (proposal_iteration_ == 0)
	generate_success = generateSegmentAddingProposal(0);
      else if (proposal_iteration_ == 1 && use_concave_hull_proposal_first_)
	generate_success = generateConcaveHullProposal(true);
      while (generate_success == false) {
	int proposal_type_index = rand() % (NUM_PROPOSAL_TYPES - 1 + current_solution_num_surfaces_ * 2);
	switch (proposal_type_index) {
	case 0:
	  generate_success = generateSegmentRefittingProposal();
	  break;
	case 1:
	  generate_success = generateConcaveHullProposal(true);
	  break;
	case 2:
	  generate_success = generateLayerSwapProposal();
	  break;
	case 3:
	  generate_success = generateSegmentAddingProposal();
	  break;
	case 4:
	  generate_success = generateBackwardMergingProposal();
	  break;
	  // case 6:
	  //   generate_success = generateBoundaryRefinementProposal();
	  //   break;
	case 5:
	  generate_success = generateStructureExpansionProposal();
	  break;
	default:
	  generate_success = generateSingleSurfaceExpansionProposal((proposal_type_index - (NUM_PROPOSAL_TYPES - 1)) % current_solution_num_surfaces_, (proposal_type_index - (NUM_PROPOSAL_TYPES - 1)) / current_solution_num_surfaces_);
	  break;
	}
      }
      LayerLabelSpace label_space;
      label_space.setLabelSpace(proposal_labels_);
      label_space.setSegments(proposal_segments_);
      label_space.setNumPixels(NUM_PIXELS_);
      label_space.setNumLayers(NUM_LAYERS_);
      proposal_label_space.unionSpace(label_space);
      proposal_iteration_++;
      if (proposal_iteration_ == 1)
        break;
    }
    return;
  }

  const bool BINARY_PROPOSAL = true;
  if (BINARY_PROPOSAL) {
    binary_proposal_designer_->setCurrentSolution(current_solution);
    for (int label_space_index = 0; label_space_index < N; label_space_index++) {
      LayerLabelSpace label_space;    
      bool generate_success = false;
      if (proposal_iteration_ == 0)
	generate_success = binary_proposal_designer_->getProposal(label_space, 0);
      else if (proposal_iteration_ == 1)
	generate_success = binary_proposal_designer_->getProposal(label_space, 1);
      while (generate_success == false) {
	int proposal_type_index = rand() % (NUM_PROPOSAL_TYPES - 1 + current_solution_num_surfaces_ * 2);
	generate_success = binary_proposal_designer_->getProposal(label_space, proposal_type_index);
      }
      //cout << label_space.getNumSegments() << endl;
      ///exit(1);
      proposal_label_space.appendSpace(label_space);
      proposal_iteration_++;
      if (proposal_iteration_ == 1)
        break;
    }
    if (N == 0)
      proposal_label_space.appendSpace(current_solution);
    return;
  }

  if (proposal_type_index_ptr_ < 0 || proposal_type_index_ptr_ >= proposal_type_indices_.size()) {
    random_shuffle(proposal_type_indices_.begin(), proposal_type_indices_.end());
    proposal_type_index_ptr_ = 0;
    all_proposal_iteration_++;
    // if (all_proposal_iteration_ > NUM_ALL_PROPOSAL_ITERATIONS_)
    //   return;
  }
  
  // if (iteration < segmentations_.size() + 2) {
  //   if (generateSegmentationProposal(iteration % segmentations_.size()) == false)
  //     return false;
  bool first_attempt = true;
  if (proposal_iteration_ == 0) {
    bool generate_success = generateSegmentAddingProposal(0);
    assert(generate_success);
  } else if (proposal_iteration_ == 1) {
    bool generate_success = generateConcaveHullProposal(true);
    assert(generate_success);
  // } else if (iteration == 2) {
  //   bool generate_success = generateSegmentAddingProposal(1);
  //   assert(generate_success);
  } else {
    while (true) {
      bool generate_success = false;
      if (single_surface_candidate_pixels_.size() > 0) {
	generate_success = generateSingleSurfaceExpansionProposal();
	// if (first_attempt && single_surface_candidate_pixels_.size() > 0)
	//   iteration--;
      } else {
	// double random_probability = randomProbability();
	// int proposal_type_index = random_probability * NUM_PROPOSAL_TYPES;
	// proposal_type_index = min(proposal_type_index, NUM_PROPOSAL_TYPES - 1);
	
	//	int proposal_type_index = rand() % NUM_PROPOSAL_TYPES;
	int proposal_type_index = proposal_type_indices_[proposal_type_index_ptr_];
	proposal_type_index_ptr_++;
	
	switch (proposal_type_index) {
	case 0:
	  generate_success = generateSegmentRefittingProposal();
	  break;
	case 1:
	  generate_success = generateConcaveHullProposal(true);
	  break;
	case 2:
	  generate_success = generateSingleSurfaceExpansionProposal();
	  break;
	  // if (randomProbability() < 1.0 / NUM_PROPOSAL_TYPES / pow(1 - 1.0 / NUM_PROPOSAL_TYPES, 3))
	  //   if (generateLayerSwapProposal() == true)
	  //     break;
	case 3:
	  generate_success = generateLayerSwapProposal();
	  break;
	case 4:
	  generate_success = generateSegmentAddingProposal();
	  break;
	case 5:
	  generate_success = generateBackwardMergingProposal();
	  break;
	  // case 6:
	//   generate_success = generateBoundaryRefinementProposal();
	//   break;
	case 6:
	  generate_success = generateStructureExpansionProposal();
	  break;
	default:
	  return;
	  // case 8:
	  //   generate_success = generateBSplineSurfaceProposal();
	  //   break;
	  // case 3:
	  //   generate_success = generateSurfaceDilationProposal();
	  //   break;

	  // case 7:
	  //   generate_success = generateInpaintingProposal();
	  //   break;
	}
      }      
      if (generate_success == true)
	break;
      first_attempt = false;
    }
  }

  proposal_label_space.setLabelSpace(proposal_labels_);
  proposal_label_space.setSegments(proposal_segments_);
  proposal_label_space.setNumPixels(NUM_PIXELS_);
  proposal_label_space.setNumLayers(NUM_LAYERS_);
  proposal_iteration_++;
  return;
}

bool ProposalDesigner::getLastProposal(vector<vector<int> > &proposal_labels, int &proposal_num_surfaces, map<int, Segment> &proposal_segments, string &proposal_type)
{
  NUM_LAYERS_++;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    current_solution_labels_[pixel] += current_solution_num_surfaces_ * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1);
  bool generate_success = generateSegmentAddingProposal();
  assert(generate_success);
  
  proposal_labels = proposal_labels_;
  proposal_num_surfaces = proposal_num_surfaces_;
  proposal_segments = proposal_segments_;
  proposal_type = proposal_type_;
  return true;
}

// string ProposalDesigner::getProposalType()
// {
//   switch (proposal_type_) {
//   case 0:
//     return "alpha expansion";
//   case 1:
//     return "connected region";
//   case 2:
//     return "convex structure";
//   }
// }

void ProposalDesigner::getInitialLabelSpace(LayerLabelSpace &label_space)
{
  getProposals(label_space, LayerLabelSpace(), 1);
  //generateEmptyRepresentationProposal();
}

void ProposalDesigner::convertProposalLabelsFormat()
{
  int dim_1 = proposal_labels_.size();
  assert(dim_1 > 0);
  int dim_2 = proposal_labels_[0].size();
  assert(dim_2 > 0);
  vector<vector<int> > new_proposal_labels(dim_2, vector<int>(dim_1));
  for (int i = 0; i < dim_1; i++)
    for (int j = 0; j < dim_2; j++)
      new_proposal_labels[j][i] = proposal_labels_[i][j];
  proposal_labels_ = new_proposal_labels;
}

void ProposalDesigner::addIndicatorVariables(const int num_indicator_variables)
{
  //int num = num_indicator_variables == -1 ? pow(NUM_SURFACES_ + 1, NUM_LAYERS_) : num_indicator_variables;
  int num = num_indicator_variables == -1 ? NUM_LAYERS_ * proposal_num_surfaces_ : num_indicator_variables;
  vector<int> indicator_labels(2);
  indicator_labels[0] = 0;
  indicator_labels[1] = 1;
  for (int i = 0; i < num; i++)
    proposal_labels_.push_back(indicator_labels);
}

void ProposalDesigner::addSegmentLayerProposals(const bool restrict_segment_in_one_layer)
{
  //int num = num_indicator_variables == -1 ? pow(NUM_SURFACES_ + 1, NUM_LAYERS_) : num_indicator_variables;
  map<int, vector<bool> > segment_layer_mask_map;
  for (int segment_id = 0; segment_id < proposal_num_surfaces_; segment_id++)
    segment_layer_mask_map[segment_id] = vector<bool>(NUM_LAYERS_, false);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposals = proposal_labels_[pixel];
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++) {
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
        int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	if (surface_id < proposal_num_surfaces_)
	  segment_layer_mask_map[surface_id][layer_index] = true;
      }
    }
  }
  
  vector<vector<int> > segment_layer_proposals(proposal_num_surfaces_);
  for (map<int, vector<bool> >::const_iterator segment_it = segment_layer_mask_map.begin(); segment_it != segment_layer_mask_map.end(); segment_it++) {
    vector<bool> layer_mask = segment_it->second;
    if (restrict_segment_in_one_layer == true) {
      segment_layer_proposals[segment_it->first].push_back(0);
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
        if (layer_mask[layer_index] == true)
	  segment_layer_proposals[segment_it->first].push_back(pow(2, NUM_LAYERS_ - 1 - layer_index));
    } else {
      for (int proposal = 0; proposal < static_cast<int>(pow(2, NUM_LAYERS_)); proposal++) {
	bool has_conflict = false;
	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	  if (layer_mask[layer_index] == 0 && proposal / static_cast<int>(pow(2, NUM_LAYERS_ - 1 - layer_index) + 0.5) % 2 == 1) {
	    has_conflict = true;
	    break;
	  }
	}
	if (has_conflict == false)
          segment_layer_proposals[segment_it->first].push_back(proposal);
      }
    }
  }
  for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++)
    if (find(segment_layer_proposals[segment_id].begin(), segment_layer_proposals[segment_id].end(), current_solution_labels_[NUM_PIXELS_ + segment_id]) == segment_layer_proposals[segment_id].end())
      segment_layer_proposals[segment_id].push_back(current_solution_labels_[NUM_PIXELS_ + segment_id]);
  
  proposal_labels_.insert(proposal_labels_.end(), segment_layer_proposals.begin(), segment_layer_proposals.end());
}

bool ProposalDesigner::checkLabelValidity(const int pixel, const int label, const int num_surfaces, const map<int, Segment> &segments)
{
  double previous_depth = 0;
  //  bool inside_ROI = ROI_mask_[pixel];
  
  bool has_depth_conflict = false;
  bool has_same_label = false;
  bool empty_background = false;
  bool segmentation_inconsistency = false;
  bool background_inconsistency = false;
  bool has_layer_estimation_conflict = false;
  bool sub_region_extended = false;
      
  int foremost_non_empty_surface_id = -1;
  vector<bool> used_surface_id_mask(num_surfaces, false);
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    int surface_id = label / static_cast<int>(pow(num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (num_surfaces + 1);
    if (surface_id == num_surfaces) {
      if (layer_index == NUM_LAYERS_ - 1)
        empty_background = true;
      continue;
    }
    double depth = segments.at(surface_id).getDepth(pixel);
    // if (layer_index == NUM_LAYERS_ - 1 && ((estimated_background_surfaces.count(segment_id) > 0 && surface_id != segment_id) || estimated_background_surfaces.count(surface_id) == 0)) {
    //   background_inconsistency = true;
    //   break;
    // }
    // if (confident_segment_layer_map.count(surface_id) > 0 && confident_segment_layer_map[surface_id] != layer_index) {
    //   has_layer_estimation_conflict = true;
    //   break;
    // }
    if (used_surface_id_mask[surface_id] == true) {
      has_same_label = true;
      break;
    }
    used_surface_id_mask[surface_id] = true;
    // if (segment_sub_region_mask[surface_id][pixel] == false) {
    //   sub_region_extended = true;
    //   break;
    // }
    if (foremost_non_empty_surface_id == -1) {
      foremost_non_empty_surface_id = surface_id;
      //      previous_depth = depth;
      
      // if (foremost_non_empty_surface_id != segment_id) {
      //   segmentation_inconsistency = true;
      //   break;
      // }
    }
    if (depth < previous_depth - statistics_.depth_conflict_threshold) {
      // if (pixel == 20803)
      // 	cout << "depth conflict: " << depth << '\t' << previous_depth << endl;
      has_depth_conflict = true;
      break;
    }
    previous_depth = depth;
  }
  if (has_depth_conflict == false && has_same_label == false && empty_background == false) // && background_inconsistency == false && has_layer_estimation_conflict == false && sub_region_extended == false && segmentation_inconsistency == false)
    return true;
  else
    return false;
}

void ProposalDesigner::writeSegmentationImage(const vector<int> &segmentation, const string filename)
{ 
  Mat segmentation_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  map<int, int> segment_center_x;
  map<int, int> segment_center_y;
  map<int, int> segment_pixel_counter;
  map<int, int> color_table;
  for (int i = 0; i < NUM_PIXELS_; i++) {
    int x = i % IMAGE_WIDTH_;
    int y = i / IMAGE_WIDTH_;

    int surface_id = segmentation[i];
    if (color_table.count(surface_id) == 0)
      color_table[surface_id] = rand() % (256 * 256 * 256);
    int surface_color = color_table[surface_id];
    segmentation_image.at<Vec3b>(y, x) = Vec3b(surface_color % 256, surface_color % 256, surface_color % 256);

    segment_center_x[surface_id] += x;
    segment_center_y[surface_id] += y;
    segment_pixel_counter[surface_id]++;
  }
  for (map<int, int>::const_iterator segment_it = segment_pixel_counter.begin(); segment_it != segment_pixel_counter.end(); segment_it++) {
    Point origin(segment_center_x[segment_it->first] / segment_it->second, segment_center_y[segment_it->first] / segment_it->second);
    char *text = new char[10];
    sprintf(text, "%d", segment_it->first);
    putText(segmentation_image, text, origin, FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 0, 255, 1));
  }
  //  stringstream segmentation_image_filename;
  //  segmentation_image_filename << "Results/segmentation_image.bmp";
  imwrite(filename.c_str(), segmentation_image);
}

void ProposalDesigner::writeSegmentationImage(const Mat &segmentation_image, const string filename)
{
  map<int, int> segment_center_x;
  map<int, int> segment_center_y;
  map<int, int> segment_pixel_counter;
  for (int i = 0; i < NUM_PIXELS_; i++) {
    int x = i % IMAGE_WIDTH_;
    int y = i / IMAGE_WIDTH_;

    Vec3b color = segmentation_image.at<Vec3b>(y, x);
    int surface_id = color[0] * 256 * 256 + color[1] * 256 + color[2];
    
    segment_center_x[surface_id] += x;
    segment_center_y[surface_id] += y;
    segment_pixel_counter[surface_id]++;
  }
  Mat image = segmentation_image.clone();
  for (map<int, int>::const_iterator segment_it = segment_pixel_counter.begin(); segment_it != segment_pixel_counter.end(); segment_it++) {
    Point origin(segment_center_x[segment_it->first] / segment_it->second, segment_center_y[segment_it->first] / segment_it->second);
    char *text = new char[10];
    sprintf(text, "%d", segment_it->first);
    putText(image, text, origin, FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 0, 255, 1));
  }
  //  stringstream segmentation_image_filename;
  //  segmentation_image_filename << "Results/segmentation_image.bmp";
  imwrite(filename.c_str(), image);
}



bool ProposalDesigner::generateSegmentRefittingProposal()
{
  cout << "generate segment refitting proposal" << endl;
  proposal_type_ = "segment_refitting_proposal";

  const int SMALL_SEGMENT_NUM_PIXELS_THRESHOLD = 10;  
  
  vector<set<int> > layer_surface_ids_vec(NUM_LAYERS_);
  map<int, map<int, vector<int> > > segment_layer_visible_pixels;
  map<int, map<int, vector<int> > > segment_layer_pixels;
  vector<bool> occluded_segment_mask(NUM_LAYERS_, false);
  vector<bool> background_segment_mask(NUM_LAYERS_, false);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	segment_layer_pixels[surface_id][layer_index].push_back(pixel);
	if (is_visible == true) {
	  segment_layer_visible_pixels[surface_id][layer_index].push_back(pixel);
          is_visible = false;
	} else
	  occluded_segment_mask[surface_id] = true;
        layer_surface_ids_vec[layer_index].insert(surface_id);
	if (layer_index == NUM_LAYERS_ - 1)
	  background_segment_mask[surface_id] = true;
      }
    }
  }

  // Mat distance_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   distance_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = current_solution_segments_[8].calcDistance2D(1.0 * (pixel % IMAGE_WIDTH_) / IMAGE_WIDTH_, 1.0 * (pixel / IMAGE_WIDTH_) / IMAGE_HEIGHT_);
  // imwrite("Test/distance_map.bmp", distance_image);
  // exit(1);
  
  proposal_segments_ = current_solution_segments_;

  int new_proposal_segment_index = current_solution_num_surfaces_;
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  //set<int> expansion_segments;
  for (map<int, map<int, vector<int> > >::const_iterator segment_it = segment_layer_pixels.begin(); segment_it != segment_layer_pixels.end(); segment_it++) {
    for (map<int, vector<int> >::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++)
      for (vector<int>::const_iterator pixel_it = layer_it->second.begin(); pixel_it != layer_it->second.end(); pixel_it++)
	layer_pixel_segment_indices_map[layer_it->first][*pixel_it].insert(segment_it->first);
    // if (segment_it->second.size() == 1)
    //   expansion_segments.insert(segment_it->first);
  }

  for (map<int, map<int, vector<int> > >::const_iterator segment_it = segment_layer_visible_pixels.begin(); segment_it != segment_layer_visible_pixels.end(); segment_it++) {
    vector<int> visible_pixels;
    
    for (map<int, vector<int> >::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++)
      visible_pixels.insert(visible_pixels.end(), layer_it->second.begin(), layer_it->second.end());

    if (visible_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
      continue;

    vector<bool> fitting_pixel_mask(NUM_PIXELS_, false);
    for (vector<int>::const_iterator pixel_it = visible_pixels.begin(); pixel_it != visible_pixels.end(); pixel_it++)
      fitting_pixel_mask[*pixel_it] = true;
    
    vector<double> depth_plane_1;
    {
      Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, visible_pixels, penalties_, statistics_);
      vector<int> fitted_pixels = segment.getSegmentPixels();
      if (fitted_pixels.size() >= SMALL_SEGMENT_NUM_PIXELS_THRESHOLD && segment.getType() >= 0) {
	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	  fitting_pixel_mask[*pixel_it] = false;

	for (map<int, vector<int> >::const_iterator layer_it = segment_layer_pixels[segment_it->first].begin(); layer_it != segment_layer_pixels[segment_it->first].end(); layer_it++)
	  for (vector<int>::const_iterator pixel_it = layer_it->second.begin(); pixel_it != layer_it->second.end(); pixel_it++)
	    layer_pixel_segment_indices_map[layer_it->first][*pixel_it].insert(new_proposal_segment_index);
	// if (segment_it->second.size() == 1)
	//   expansion_segments.insert(new_proposal_segment_index);
	
	proposal_segments_[new_proposal_segment_index] = segment;
	new_proposal_segment_index++;

	depth_plane_1 = segment.getDepthPlane();
      }
    }
    
    {
      vector<int> fitting_pixels;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
        if (fitting_pixel_mask[pixel] == true)
          fitting_pixels.push_back(pixel);

      Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, fitting_pixels, penalties_, statistics_);
      vector<int> fitted_pixels = segment.getSegmentPixels();
      if (fitted_pixels.size() >= SMALL_SEGMENT_NUM_PIXELS_THRESHOLD && segment.getType() >= 0) {
	vector<double> depth_plane_2 = segment.getDepthPlane();

	double cos_value = 0;
	for (int c = 0; c < 3; c++)
	  cos_value += depth_plane_1[c] * depth_plane_2[c];
	double angle = acos(min(abs(cos_value), 1.0));
	if (angle > statistics_.similar_angle_threshold) {
	  for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	    layer_pixel_segment_indices_map[segment_it->second.begin()->first][*pixel_it].insert(new_proposal_segment_index);
	  //expansion_segments.insert(new_proposal_segment_index);

	  proposal_segments_[new_proposal_segment_index] = segment;
	  new_proposal_segment_index++;
	}
      }
    }

    if (background_segment_mask[segment_it->first] == false) {
      if (visible_pixels.size() <= statistics_.bspline_surface_num_pixels_threshold) {
        Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, visible_pixels, penalties_, statistics_, 2);
	vector<int> fitted_pixels = segment.getSegmentPixels();
        if (fitted_pixels.size() >= SMALL_SEGMENT_NUM_PIXELS_THRESHOLD && segment.getType() >= 0) {
	  for (map<int, vector<int> >::const_iterator layer_it = segment_layer_pixels[segment_it->first].begin(); layer_it != segment_layer_pixels[segment_it->first].end(); layer_it++)
	    for (vector<int>::const_iterator pixel_it = layer_it->second.begin(); pixel_it != layer_it->second.end(); pixel_it++)
	      layer_pixel_segment_indices_map[layer_it->first][*pixel_it].insert(new_proposal_segment_index);
	  // if (segment_it->second.size() == 1)
	  //   expansion_segments.insert(new_proposal_segment_index);

	  proposal_segments_[new_proposal_segment_index] = segment;
	  new_proposal_segment_index++;
	}
      }
    }
  }
  
  //   if (segment_it->second.size() == 1) {
  //     vector<int> fitting_pixels;
  //     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //       if (fitting_pixel_mask[pixel] == true)
  //         fitting_pixels.push_back(pixel);

  //     Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, fitting_pixels, penalties_, statistics_);
  //     vector<int> fitted_pixels = segment.getSegmentPixels();
  //     if (fitted_pixels.size() >= SMALL_SEGMENT_NUM_PIXELS_THRESHOLD) {
  // 	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
  // 	  layer_pixel_segment_indices_map[segment_it->second.begin()->first][*pixel_it].insert(new_proposal_segment_index);
  // 	expansion_segments.insert(new_proposal_segment_index);

  // 	proposal_segments_[new_proposal_segment_index] = segment;
  // 	new_proposal_segment_index++;
  //     }
  //   } else {
  //     for (map<int, vector<int> >::const_iterator layer_it = segment_layer_pixels[segment_it->first].begin(); layer_it != segment_layer_pixels[segment_it->first].end(); layer_it++) {
  // 	vector<int> fitting_pixels = layer_it->second;
  // 	Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, fitting_pixels, penalties_, statistics_);
  // 	vector<int> fitted_pixels = segment.getSegmentPixels();
  // 	if (fitted_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
  // 	  continue;

  // 	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
  // 	  layer_pixel_segment_indices_map[layer_it->first][*pixel_it].insert(new_proposal_segment_index);
  // 	expansion_segments.insert(new_proposal_segment_index);

  // 	proposal_segments_[new_proposal_segment_index] = segment;
  // 	new_proposal_segment_index++;
  //     }
  //   }
  // }
  
  const int NUM_DILATION_ITERATIONS = 2;
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
    while (true) {
      bool has_change = false;
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        vector<int> neighbor_pixels;
        int x = pixel % IMAGE_WIDTH_;
        int y = pixel / IMAGE_WIDTH_;
        if (x > 0)
          neighbor_pixels.push_back(pixel - 1);
        if (x < IMAGE_WIDTH_ - 1)
          neighbor_pixels.push_back(pixel + 1);
        if (y > 0)
          neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
        if (y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
        if (x > 0 && y > 0)
          neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
        if (x > 0 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y > 0)
          neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	    // if (expansion_segments.count(*segment_it) == 0)
	    //   continue;
            if (proposal_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it) && dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
              dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
              has_change = true;
            }
          }
        }
      }
      if (has_change == false)
        break;
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
    }

    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        // if (pixel_segment_indices_map[pixel].size() == 0)
        //   continue;
        vector<int> neighbor_pixels;
        int x = pixel % IMAGE_WIDTH_;
        int y = pixel / IMAGE_WIDTH_;
        if (x > 0)
          neighbor_pixels.push_back(pixel - 1);
        if (x < IMAGE_WIDTH_ - 1)
          neighbor_pixels.push_back(pixel + 1);
        if (y > 0)
          neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
        if (y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
        if (x > 0 && y > 0)
          neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
        if (x > 0 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y > 0)
          neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
          // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
          //   continue;
          for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
              continue;
            if (proposal_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
              dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
          }
        }
      }
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
    }

    layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  }
  
  Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  map<int, Vec3b> color_table;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    //    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    int layer_index = 0;
    // if (layer_pixel_segment_indices_map[layer_index][pixel].size() == 0)
    //   continue;
    int segment_index = 1;
    for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++)
      if (*segment_it >= current_solution_num_surfaces_)
	segment_index *= (*segment_it + 1);
    segment_index = layer_pixel_segment_indices_map[1][pixel].count(8) > 0 ? 1 : 0;
    if (color_table.count(segment_index) == 0) {
      Vec3b color;
      for (int c = 0; c < 3; c++)
	color[c] = rand() % 256;
      color_table[segment_index] = color;
    }
    new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_index];
      
  }
  imwrite("Test/refitted_segment_image.bmp", new_segment_image);

  // for (map<int, Segment>::const_iterator segment_it = proposal_segments_.begin(); segment_it != proposal_segments_.end(); segment_it++)
  //   cout << segment_it->first << '\t' << segment_it->second.getType() << '\t' << segment_it->second.getSegmentPixels().size() << endl;
  
  // vector<vector<set<int> > > pixel_layer_surface_ids_map(NUM_PIXELS_, vector<set<int> >(NUM_LAYERS_));
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   int current_solution_label = current_solution_labels_[pixel];
  //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //     for (set<int>::const_iterator segment_it = layer_surface_ids_vec[layer_index].begin(); segment_it != layer_surface_ids_vec[layer_index].end(); segment_it++)
  // 	if (proposal_segments_.at(*segment_it).checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel))
  // 	  pixel_layer_surface_ids_map[pixel][layer_index].insert(*segment_it);
  //     int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
  //     if (surface_id < current_solution_num_surfaces_)
  // 	break;
  //   }
  // }

  proposal_num_surfaces_ = proposal_segments_.size();
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  
  //  int max_num_proposals = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    map<int, set<int> > pixel_layer_surfaces_map;
    //vector<set<int> > layer_surface_ids_map = pixel_layer_surface_ids_map[pixel];
    //vector<int> layer_segment_index_map = pixel_layer_segment_index_map[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	pixel_layer_surfaces_map[layer_index].insert(surface_id);
	// if (layer_segment_index_map[layer_index] != -1)
        //   pixel_layer_surfaces_map[layer_index].insert(layer_segment_index_map[layer_index]);
	// else
	//   pixel_layer_surfaces_map[layer_index].insert(segment_new_segments_map[surface_id].begin(), segment_new_segments_map[surface_id].end());
      } else {
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
        // for (set<int>::const_iterator surface_it = layer_surface_ids_map[layer_index].begin(); surface_it != layer_surface_ids_map[layer_index].end(); surface_it++)
        //   //if (proposal_segments_[*surface_it].checkPixelFitting(point_cloud_, normals_, pixel))
	//   pixel_layer_surfaces_map[layer_index].insert(*surface_it);
	// //	pixel_layer_surfaces_map[layer_index].insert(layer_surface_ids_map[layer_index].begin(), layer_surface_ids_map[layer_index].end());
      }
    }

    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
    
    for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    // if (pixel == 11084) {
    //   for (set<int>::const_iterator segment_it = segment_new_segments_map[1].begin(); segment_it != segment_new_segments_map[1].end(); segment_it++)
    // 	cout << *segment_it << endl;
    //   for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++) {
    //  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //    int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //    cout << proposal_surface_id << '\t';
    //  }
    //  cout << endl;
    //   }
    //   exit(1);
    // }
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
      //   cout << pixel_proposal[proposal_index] << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;


    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }

    // if (valid_pixel_proposals.size() > max_num_proposals) {
    //   cout << "max number of proposals: " << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << valid_pixel_proposals.size() << endl;
    //   max_num_proposals = valid_pixel_proposals.size();
    // }    
  }

  addIndicatorVariables();
  return true;
}

// bool ProposalDesigner::generateBSplineSurfaceProposal()
// {
//   cout << "generate bspline surface proposal" << endl;
//   proposal_type_ = "bspline_surface_proposal";

//   map<int, vector<int> > segment_visible_pixels;
//   vector<set<int> > layer_surface_ids_vec(NUM_LAYERS_);
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       if (surface_id < current_solution_num_surfaces_) {
//         segment_visible_pixels[surface_id].push_back(pixel);
// 	layer_surface_ids_vec[layer_index].insert(surface_id);
// 	layer_surface_ids_vec[layer_index].insert(surface_id + current_solution_num_surfaces_);
//         break;
//       }
//     }
//   }

//   proposal_segments_ = current_solution_segments_;  

//   //int new_proposal_segment_index = current_solution_num_surfaces_;
//   for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
//     //    proposal_segments_[segment_it->first + current_solution_num_surfaces_] = segment_it->second;
//     proposal_segments_[segment_it->first + current_solution_num_surfaces_] = Segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, segment_visible_pixels[segment_it->first], penalties_, statistics_, 2);
//   }
  
//   //vector<vector<int> > pixel_layer_segment_index_map(NUM_PIXELS_, vector<int>(NUM_LAYERS_, -1));
  
//   // for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//   //   for (set<int>::const_iterator segment_it = layer_surface_ids_vec[layer_index].begin(); segment_it != layer_surface_ids_vec[layer_index].end(); segment_it++) {
//   //     Segment new_segment = current_solution_segments_[*segment_it];
//   //     new_segment.fitBSplineSurface(image_, point_cloud_, normals_, visible_pixels);
//   //     proposal_segments_[new_proposal_segment_index] = new_segment;
//   //     layer_surface_ids_vec[layer_index].insert(new_proposal_segment_index);
//   //     new_proposal_segment_index++;
//   //     //if (new_segment.buildSubSegment(image_, point_cloud_, normals_, segment_visible_pixels[ *segment_it])) {
//   //     // 	proposal_segments_[new_proposal_segment_index] = new_segment;
//   //     // 	layer_surface_ids_vec[layer_index].insert(new_proposal_segment_index);
//   //     // }
//   //   }
//   // }
  
//   // Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
//   // map<int, Vec3b> color_table;
//   // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//   //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//   //     // int segment_index = pixel_layer_segment_index_map[pixel][layer_index];
//   //     // if (segment_index == -1)
//   //     //   continue;
//   //     if (color_table.count(segment_index) == 0) {
//   //       Vec3b color;
//   //       for (int c = 0; c < 3; c++)
//   //         color[c] = rand() % 256;
//   //       color_table[segment_index] = color;
//   //     }
//   //     new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_index];
//   //   }
//   // }
//   // imwrite("Test/refitted_segment_image.bmp", new_segment_image);

  
//   // vector<vector<set<int> > > pixel_layer_surface_ids_map(NUM_PIXELS_, vector<set<int> >(NUM_LAYERS_));
//   // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//   //   int current_solution_label = current_solution_labels_[pixel];
//   //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//   //     for (set<int>::const_iterator segment_it = layer_surface_ids_vec[layer_index].begin(); segment_it != layer_surface_ids_vec[layer_index].end(); segment_it++)
//   //       if (proposal_segments_.at(*segment_it).checkPixelFitting(point_cloud_, normals_, pixel))
//   //         pixel_layer_surface_ids_map[pixel][layer_index].insert(*segment_it);
//   //     int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//   //     if (surface_id < current_solution_num_surfaces_)
//   //       break;
//   //   }
//   // }

//   proposal_num_surfaces_ = proposal_segments_.size();
  
//   proposal_labels_.assign(NUM_PIXELS_, vector<int>());
//   current_solution_indices_.assign(NUM_PIXELS_, 0);
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];
//     map<int, set<int> > pixel_layer_surfaces_map;
//     //vector<set<int> > layer_surface_ids_map = pixel_layer_surface_ids_map[pixel];
//     //vector<int> layer_segment_index_map = pixel_layer_segment_index_map[pixel];
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       if (surface_id < current_solution_num_surfaces_) {
//         pixel_layer_surfaces_map[layer_index].insert(surface_id);
// 	pixel_layer_surfaces_map[layer_index].insert(surface_id + current_solution_num_surfaces_);
//       } else {
//         pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
// 	for (set<int>::const_iterator surface_it = layer_surface_ids_vec[layer_index].begin(); surface_it != layer_surface_ids_vec[layer_index].end(); surface_it++)
// 	  if (proposal_segments_[*surface_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel))
// 	    pixel_layer_surfaces_map[layer_index].insert(*surface_it);
//       }
//     }
    
//     vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

//     // if (pixel == 11084) {
//     //   for (set<int>::const_iterator segment_it = segment_new_segments_map[1].begin(); segment_it != segment_new_segments_map[1].end(); segment_it++)
//     //  cout << *segment_it << endl;
//     //   for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++) {
//     //  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//     //    int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
//     //    cout << proposal_surface_id << '\t';
//     //  }
//     //  cout << endl;
//     //   }
//     //   exit(1);
//     // }
    
//     vector<int> valid_pixel_proposals;
//     for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
//       if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
//         valid_pixel_proposals.push_back(*label_it);

//     if (valid_pixel_proposals.size() == 0) {
//       cout << "empty proposal at pixel: " << pixel << endl;
//       // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
//       //   cout << pixel_proposal[proposal_index] << endl;
//       exit(1);
//     }      

//     proposal_labels_[pixel] = valid_pixel_proposals;


//     if (current_solution_num_surfaces_ > 0) {
//       current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
//       if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
//         cout << "has no current solution label at pixel: " << pixel << endl;
//         exit(1);
//       }
//     }
//   }

//   //addSegmentLayerProposals(false);
//   addIndicatorVariables();
//   return true;
// }

// bool ProposalDesigner::generateSegmentRefittingProposal()
// {
//   cout << "generate refitting proposal" << endl;
//   proposal_type_ = "refitting_proposal";
  
//   map<int, vector<int> > segment_visible_pixels;
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       if (surface_id < current_solution_num_surfaces_) {
//         segment_visible_pixels[surface_id].push_back(pixel);
//         break;
//       }
//     }
//   }
//   proposal_segments_.clear();  
//   for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
//     proposal_segments_[segment_it->first] = segment_it->second;
//     proposal_segments_[segment_it->first + current_solution_num_surfaces_] = segment_it->second;
//     proposal_segments_[segment_it->first + current_solution_num_surfaces_].refitSegment(image_, point_cloud_, segment_visible_pixels[segment_it->first]);
//   }
//   proposal_num_surfaces_ = proposal_segments_.size();
  
//   proposal_labels_.assign(NUM_PIXELS_, vector<int>());
//   current_solution_indices_.assign(NUM_PIXELS_, 0);
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];
//     map<int, set<int> > pixel_layer_surfaces_map;
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       if (surface_id < current_solution_num_surfaces_) {
// 	pixel_layer_surfaces_map[layer_index].insert(surface_id);
// 	for (int target_layer_index = 0; target_layer_index < NUM_LAYERS_; target_layer_index++)
// 	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id + current_solution_num_surfaces_);
//       } else
// 	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
//     }
    
//     vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

//     vector<int> valid_pixel_proposals;
//     for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
//       if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
//         valid_pixel_proposals.push_back(*label_it);

//     if (valid_pixel_proposals.size() == 0) {
//       cout << "empty proposal at pixel: " << pixel << endl;
//       // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
//       //   cout << pixel_proposal[proposal_index] << endl;
//       exit(1);
//     }      

//     proposal_labels_[pixel] = valid_pixel_proposals;


//     if (current_solution_num_surfaces_ > 0) {
//       current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
//       if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
//         cout << "has no current solution label at pixel: " << pixel << endl;
//         exit(1);
//       }
//     }
//   }

//   addIndicatorVariables();
//   return true;
// }

 bool ProposalDesigner::generateSingleSurfaceExpansionProposal(const int denoted_expansion_segment_id, const int denoted_expansion_type)
{
  cout << "generate single surface expansion proposal" << endl;
  proposal_type_ = "single_surface_expansion_proposal";

  if (single_surface_candidate_pixels_.size() == 0) {
    single_surface_candidate_pixels_.assign(NUM_PIXELS_ * 2, -1);
    for (int pixel = 0; pixel < NUM_PIXELS_ * 2; pixel++)
      single_surface_candidate_pixels_[pixel] = pixel;
  }
  
  int expansion_segment_id = denoted_expansion_segment_id;
  int expansion_type = denoted_expansion_type;
  if (current_solution_segments_.count(expansion_segment_id) == 0 || expansion_type == -1) {
    //    int random_pixel = rand() % NUM_PIXELS_;
    int random_pixel = single_surface_candidate_pixels_[rand() % single_surface_candidate_pixels_.size()];
    int current_solution_label = current_solution_labels_[random_pixel % NUM_PIXELS_];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	expansion_segment_id = surface_id;
	break;
      }
    }
    expansion_type = random_pixel / NUM_PIXELS_;
    
    // map<int, double> segment_confidence_map;
    // double confidence_sum = 0;
    // for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
    //   double confidence = segment_it->second.getConfidence();
    //   segment_confidence_map[segment_it->first] = confidence;
    //   confidence_sum += confidence;
    // }

    // double selected_confidence = randomProbability() * confidence_sum;
    // confidence_sum = 0;
  
    // for (map<int, double>::const_iterator segment_it = segment_confidence_map.begin(); segment_it != segment_confidence_map.end(); segment_it++) {
    //   confidence_sum += segment_it->second;
    //   if (confidence_sum >= selected_confidence) {
    // 	expansion_segment_id = segment_it->first;
    // 	break;
    //   }
    // }
    // assert(expansion_segment_id != -1);
  }

  
  // int num_attempts = 0;
  // while (true) {
  //   if (num_attempts >= current_solution_num_surfaces_)
  //     return false;
  //   num_attempts++;
  //   expansion_segment_id = segment_id == -1 ? rand() % current_solution_num_surfaces_ : segment_id;

  //   if (current_solution_segments_.count(expansion_segment_id) == 0)
  //     continue;
  
  //   if (current_solution_segments_[expansion_segment_id].getConfidence() < 0.5)
  //     continue;
  //   break;
  // }

  map<int, int> expansion_segment_layer_counter;
  bool is_occluded = false;
  vector<bool> expansion_segment_visible_pixel_mask(NUM_PIXELS_, false);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id == expansion_segment_id) {
        expansion_segment_layer_counter[layer_index]++;
	if (is_visible == false) {
	  is_occluded = true;
	  break;
	} else
	  expansion_segment_visible_pixel_mask[pixel] = true;
      }
      if (surface_id < current_solution_num_surfaces_)
	is_visible = false;
    }
  }
  vector<int> new_single_surface_candidate_pixels;
  for (vector<int>::const_iterator pixel_it = single_surface_candidate_pixels_.begin(); pixel_it != single_surface_candidate_pixels_.end(); pixel_it++)
    if (*pixel_it / NUM_PIXELS_ != expansion_type || expansion_segment_visible_pixel_mask[*pixel_it % NUM_PIXELS_] == false)
      new_single_surface_candidate_pixels.push_back(*pixel_it);
  single_surface_candidate_pixels_ = new_single_surface_candidate_pixels;
  

  if (expansion_segment_layer_counter.size() > 1 && expansion_type == 1)
    return false;
  if (is_occluded && expansion_type == 0)
    return false;

  // if (expansion_segment_layer_counter.size() > 1)
  //   expansion_type = 0;
  // else if (is_occluded == true)
  //   expansion_type = 1;
  // else
  //   expansion_type = randomProbability() < 0.5 ? 0 : 1;
  
  //  int expansion_segment_layer_index = expansion_segment_layer_counter.begin()->first;
    
  int expansion_segment_layer_index = -1;
  int max_layer_count = 0;
  for (map<int, int>::const_iterator layer_it = expansion_segment_layer_counter.begin(); layer_it != expansion_segment_layer_counter.end(); layer_it++) {
    if (layer_it->second > max_layer_count) {
      expansion_segment_layer_index = layer_it->first;
      max_layer_count = layer_it->second;
    }
  }
  if (expansion_segment_layer_index == -1)
    return false;

  
  cout << "segment: " << expansion_segment_id << "\texpansion type: " << expansion_type << endl;
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    if (expansion_type == 0) {
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
	pixel_layer_surfaces_map[layer_index].insert(expansion_segment_id);
      for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++)
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    } else {
      pixel_layer_surfaces_map[expansion_segment_layer_index].insert(expansion_segment_id);
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - expansion_segment_layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_ && surface_id != expansion_segment_id)
	for (int target_layer_index = 0; target_layer_index < expansion_segment_layer_index; target_layer_index++)
	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);
      for (int target_layer_index = 0; target_layer_index < expansion_segment_layer_index; target_layer_index++)
	pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);
    }
        
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
	valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
	cout << "has no current solution label at pixel: " << pixel << endl;
	exit(1);
      }
    }


    // if (pixel == 132 * IMAGE_WIDTH_ + 57) {
    //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
    // 	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    // 	  int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    // 	  cout << proposal_surface_id << '\t';
    // 	}
    // 	cout << endl;
    //   }
    //   exit(1);
    // }
  }

  //addSegmentLayerProposals(true);
  addIndicatorVariables();

  return true;
}

bool ProposalDesigner::generateLayerSwapProposal()
{
  cout << "generate layer swap proposal" << endl;
  proposal_type_ = "layer_swap_proposal";


  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_)
        layer_pixel_segment_indices_map[layer_index][pixel].insert(surface_id);
    }
  }

  const int NUM_DILATION_ITERATIONS = 2;
  for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
    vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
    
    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        // if (pixel_segment_indices_map[pixel].size() == 0)
        //   continue;
        vector<int> neighbor_pixels;
        int x = pixel % IMAGE_WIDTH_;
        int y = pixel / IMAGE_WIDTH_;
        if (x > 0)
          neighbor_pixels.push_back(pixel - 1);
        if (x < IMAGE_WIDTH_ - 1)
          neighbor_pixels.push_back(pixel + 1);
        if (y > 0)
          neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
        if (y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
        if (x > 0 && y > 0)
          neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
        if (x > 0 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y > 0)
          neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
          // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
          //   continue;
          for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
              continue;
            if (current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
              new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
          }
        }
      }
      pixel_segment_indices_map = new_pixel_segment_indices_map;
    }
    layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  }

  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;

  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];    

    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
      if (layer_index < NUM_LAYERS_ - 1)
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }
    for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
      set<int> segments = layer_pixel_segment_indices_map[layer_index][pixel];
      for (int target_layer_index = max(0, layer_index - 1); target_layer_index <= min(NUM_LAYERS_ - 1, layer_index + 1); target_layer_index++)
	pixel_layer_surfaces_map[target_layer_index].insert(segments.begin(), segments.end());
    }
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);    

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
      //   cout << pixel_proposal[proposal_index] << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }

    
    // if (pixel == 134 * IMAGE_WIDTH_ + 155) {
    //   cout << current_solution_segments_[12].getDepth(pixel) << '\t' << current_solution_segments_[1].getDepth(pixel) << endl;
    //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
    //     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //       int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //       cout << proposal_surface_id << '\t';
    //     }
    //     cout << endl;
    //   }
    //   exit(1);
    // }
  }

  //addSegmentLayerProposals(true);
  addIndicatorVariables();
  

  // proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   proposal_labels_[pixel].push_back(current_solution_labels_[pixel]);

  // current_solution_indices_.assign(NUM_PIXELS_, 0);
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   int current_solution_label = current_solution_labels_[pixel];    

  //   vector<int> neighbor_pixels;
  //   int x = pixel % IMAGE_WIDTH_;
  //   int y = pixel / IMAGE_WIDTH_;
  //   if (x > 0)
  //     neighbor_pixels.push_back(pixel - 1);
  //   if (x < IMAGE_WIDTH_ - 1)
  //     neighbor_pixels.push_back(pixel + 1);
  //   if (y > 0)
  //     neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
  //   if (y < IMAGE_HEIGHT_ - 1)
  //     neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
  //   if (x > 0 && y > 0)
  //     neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
  //   if (x > 0 && y < IMAGE_HEIGHT_ - 1)
  //     neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
  //   if (x < IMAGE_WIDTH_ - 1 && y > 0)
  //     neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
  //   if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
  //     neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);

  //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //     int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
  //     if (surface_id == current_solution_num_surfaces_)
  // 	continue;
  //     for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
  // 	int current_solution_neighbor_label = current_solution_labels_[*neighbor_pixel_it];
  // 	int neighbor_surface_id = current_solution_neighbor_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
  // 	if (neighbor_surface_id == surface_id)
  // 	  continue;
  // 	int neighbor_proposal_label = current_solution_neighbor_label + (surface_id - neighbor_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index);
  // 	vector<int> neighbor_pixel_proposals = proposal_labels_[*neighbor_pixel_it];
  // 	if (find(neighbor_pixel_proposals.begin(), neighbor_pixel_proposals.end(), neighbor_proposal_label) == neighbor_pixel_proposals.end()) {
  // 	  if (checkLabelValidity(*neighbor_pixel_it, neighbor_proposal_label, proposal_num_surfaces_, proposal_segments_) == true) {
  // 	    neighbor_pixel_proposals.push_back(neighbor_proposal_label);
  // 	    proposal_labels_[*neighbor_pixel_it] = neighbor_pixel_proposals;
  // 	  }
  // 	}
  //     }
  //   }

  //   current_solution_indices_[pixel] = 0;
  // }

  //  addIndicatorVariables();

  return true;
}

// bool ProposalDesigner::generateLayerSwapProposal()
// {
//   cout << "generate layer swap proposal" << endl;
//   proposal_type_ = "layer_swap_proposal";

//   proposal_num_surfaces_ = current_solution_num_surfaces_;
//   proposal_segments_ = current_solution_segments_;

//   proposal_labels_.assign(NUM_PIXELS_, vector<int>());
//   current_solution_indices_.assign(NUM_PIXELS_, 0);
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];

//     map<int, set<int> > pixel_layer_surfaces_map;
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       if (surface_id < current_solution_num_surfaces_)
// 	for (int target_layer_index = 0; target_layer_index < NUM_LAYERS_; target_layer_index++)
// 	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);

//       if (layer_index < NUM_LAYERS_ - 1)
//         pixel_layer_surfaces_map[layer_index].insert(current_solution_num_surfaces_);
//     }    

//     vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);    

//     vector<int> valid_pixel_proposals;
//     for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
//       if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
// 	valid_pixel_proposals.push_back(*label_it);

//     if (valid_pixel_proposals.size() == 0) {
//       cout << "empty proposal at pixel: " << pixel << endl;
//       // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
//       //   cout << pixel_proposal[proposal_index] << endl;
//       exit(1);
//     }      

//     proposal_labels_[pixel] = valid_pixel_proposals;
    
//     if (current_solution_num_surfaces_ > 0) {
//       current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
//       if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
//         cout << "has no current solution label at pixel: " << pixel << endl;
//         exit(1);
//       }
//     }
//   }

//   //addSegmentLayerProposals(true);
//   addIndicatorVariables();

//   return true;
// }

bool ProposalDesigner::generateConcaveHullProposal(const bool consider_background)
{
  cout << "generate concave hull proposal" << endl;
  proposal_type_ = "concave_hull_proposal";
  

  vector<vector<int>> layer_pixel_inpainting_surface_ids(NUM_LAYERS_);
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    vector<int> pixel_inpainting_surface_ids(NUM_PIXELS_, -1);
    if (layer_index == 0) { // || (consider_background == false && layer_index == NUM_LAYERS_ - 1)) {
      layer_pixel_inpainting_surface_ids[layer_index] = pixel_inpainting_surface_ids;
      continue;
    }

    vector<int> layer_surface_ids(NUM_PIXELS_);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int current_solution_label = current_solution_labels_[pixel];
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      layer_surface_ids[pixel] = surface_id;
    }

    vector<bool> visited_pixel_mask(NUM_PIXELS_, false);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      if (visited_pixel_mask[pixel] == true || layer_surface_ids[pixel] == current_solution_num_surfaces_)
        continue;
      vector<bool> region_mask(NUM_PIXELS_, false);
      vector<int> border_pixels;
      border_pixels.push_back(pixel);
      visited_pixel_mask[pixel] = true;
      while (true) {
        vector<int> new_border_pixels;
        for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
          region_mask[*border_pixel_it] = true;
          vector<int> neighbor_pixels;
          int x = *border_pixel_it % IMAGE_WIDTH_;
          int y = *border_pixel_it / IMAGE_WIDTH_;
          if (x > 0)
            neighbor_pixels.push_back(*border_pixel_it - 1);
          if (x < IMAGE_WIDTH_ - 1)
            neighbor_pixels.push_back(*border_pixel_it + 1);
          if (y > 0)
            neighbor_pixels.push_back(*border_pixel_it - IMAGE_WIDTH_);
          if (y < IMAGE_HEIGHT_ - 1)
            neighbor_pixels.push_back(*border_pixel_it + IMAGE_WIDTH_);
          if (x > 0 && y > 0)
            neighbor_pixels.push_back(*border_pixel_it - 1 - IMAGE_WIDTH_);
          if (x > 0 && y < IMAGE_HEIGHT_ - 1)
            neighbor_pixels.push_back(*border_pixel_it - 1 + IMAGE_WIDTH_);
          if (x < IMAGE_WIDTH_ - 1 && y > 0)
            neighbor_pixels.push_back(*border_pixel_it + 1 - IMAGE_WIDTH_);
          if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
            neighbor_pixels.push_back(*border_pixel_it + 1 + IMAGE_WIDTH_);
          for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
            if (layer_surface_ids[*neighbor_pixel_it] != current_solution_num_surfaces_ && visited_pixel_mask[*neighbor_pixel_it] == false) {
	      new_border_pixels.push_back(*neighbor_pixel_it);
	      visited_pixel_mask[*neighbor_pixel_it] = true;	
  	    }
          }
	}
        if (new_border_pixels.size() == 0)
          break;
        border_pixels = new_border_pixels;
      }

      unique_ptr<ConcaveHullFinder> concave_hull_finder(new ConcaveHullFinder(IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, layer_surface_ids, current_solution_segments_, region_mask, penalties_, statistics_, consider_background));
      vector<int> concave_hull = concave_hull_finder->getConcaveHull();
      if (concave_hull.size() == 0)
	continue;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	if (region_mask[pixel] == true)
	  pixel_inpainting_surface_ids[pixel] = concave_hull[pixel];
    }
    layer_pixel_inpainting_surface_ids[layer_index] = pixel_inpainting_surface_ids;
  }

  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;

  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  int max_num_proposals = 0;  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
      if (layer_pixel_inpainting_surface_ids[layer_index][pixel] != -1 && layer_pixel_inpainting_surface_ids[layer_index][pixel] != surface_id) {
      	pixel_layer_surfaces_map[layer_index].insert(layer_pixel_inpainting_surface_ids[layer_index][pixel]);
	for (int target_layer_index = 0; target_layer_index < layer_index; target_layer_index++)
	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);
      }
    }
    for (int target_layer_index = 0; target_layer_index < NUM_LAYERS_ - 1; target_layer_index++)
      pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);
    // pixel_layer_surfaces_map[1].erase(3);
    // pixel_layer_surfaces_map[0].erase(4);
    // pixel_layer_surfaces_map[0].erase(6);
    
    // pixel_layer_surfaces_map[NUM_LAYERS_ - 1].clear();
    // layer_pixel_inpainting_surface_ids[NUM_LAYERS_ - 1].insert(pixel);
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    // if (pixel == 20803) {
    //   for (map<int, set<int> >::const_iterator layer_it = pixel_layer_surfaces_map.begin(); layer_it != pixel_layer_surfaces_map.end(); layer_it++)
    //     for (set<int>::const_iterator segment_it = layer_it->second.begin(); segment_it != layer_it->second.end(); segment_it++)
    // 	  cout << layer_it->first << '\t' << *segment_it << endl;
    //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
    //     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //       int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //       cout << proposal_surface_id << '\t';
    //     }
    //     cout << endl;
    //   }
    //   cout << proposal_num_surfaces_ << '\t' << NUM_LAYERS_ << endl;
    //   cout << current_solution_label << '\t' << valid_pixel_proposals[0] << '\t' << valid_pixel_proposals[1] << endl;
    //   exit(1);
    // }

    // if (valid_pixel_proposals.size() > 1)
    //   cout << "yes" << endl;
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
      //   cout << pixel_proposal[proposal_index] << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
    
    // if (valid_pixel_proposals.size() > max_num_proposals) {
    //   cout << "max number of proposals: " << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << valid_pixel_proposals.size() << endl;
    //   max_num_proposals = valid_pixel_proposals.size();
    // }    

    // pixel_proposals.push_back(current_solution_label);

    // for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
    //   int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
    //   if (surface_id != current_solution_num_surfaces_) {
    //  int proposal_label = current_solution_label + (current_solution_num_surfaces_ - surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index);
    //  pixel_proposals.push_back(proposal_label);
    //   }
    // }
    // proposal_labels_[pixel] = pixel_proposals;
  }

  //addSegmentLayerProposals(true);
  addIndicatorVariables();
  return true;
}

// bool ProposalDesigner::generateCleanUpProposal()
// {
//   cout << "generate clean up proposal" << endl;
//   proposal_type_ = "clean_up_proposal";

//   const double COMMON_BOUNDARY_LENGTH_THRESHOLD_RATIO = 0.1;
//   vector<vector<set<int> > > layer_pixel_inpainting_surface_ids(NUM_LAYERS_);
//   vector<set<int> > layer_surface_ids_vec(NUM_LAYERS_);
//   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//     vector<int> layer_surface_ids(NUM_PIXELS_);
//     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//       int current_solution_label = current_solution_labels_[pixel];
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       if (surface_id < current_solution_num_surfaces_)
// 	layer_surface_ids_vec[layer_index].insert(surface_id);
//       layer_surface_ids[pixel] = surface_id;

//       // if (surface_id == current_solution_num_surfaces_) {
//       // 	for (int other_layer_index = 0; other_layer_index < NUM_LAYERS_; other_layer_index++) {
//       // 	  if (other_layer_index == layer_index)
//       // 	    continue;
//       // 	  int other_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - other_layer_index)) % (current_solution_num_surfaces_ + 1);
//       // 	  other_layer_surface_ids[pixel] = other_surface_id;
//       // 	}
//       // }
//     }

//     vector<set<int> > pixel_inpainting_surface_ids(NUM_PIXELS_);
//     vector<bool> visited_pixel_mask(NUM_PIXELS_, false);
//     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//       if (visited_pixel_mask[pixel] == true)
// 	continue;
//       int segment_id = layer_surface_ids[pixel];
//       if (segment_id == current_solution_num_surfaces_)
// 	continue;
//       vector<int> segment_pixels;
//       vector<int> border_pixels;
//       border_pixels.push_back(pixel);
//       visited_pixel_mask[pixel] = true;
//       map<int, int> neighbor_segment_counter;
//       int boundary_length = 0;
//       while (true) {
// 	vector<int> new_border_pixels;
// 	for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
// 	  segment_pixels.push_back(*border_pixel_it);
// 	  vector<int> neighbor_pixels;
// 	  int x = *border_pixel_it % IMAGE_WIDTH_;
// 	  int y = *border_pixel_it / IMAGE_WIDTH_;
// 	  if (x > 0)
// 	    neighbor_pixels.push_back(*border_pixel_it - 1);
// 	  if (x < IMAGE_WIDTH_ - 1)
// 	    neighbor_pixels.push_back(*border_pixel_it + 1);
// 	  if (y > 0)
// 	    neighbor_pixels.push_back(*border_pixel_it - IMAGE_WIDTH_);
// 	  if (y < IMAGE_HEIGHT_ - 1)
// 	    neighbor_pixels.push_back(*border_pixel_it + IMAGE_WIDTH_);
// 	  if (x > 0 && y > 0)
// 	    neighbor_pixels.push_back(*border_pixel_it - 1 - IMAGE_WIDTH_);
// 	  if (x > 0 && y < IMAGE_HEIGHT_ - 1)
// 	    neighbor_pixels.push_back(*border_pixel_it - 1 + IMAGE_WIDTH_);
// 	  if (x < IMAGE_WIDTH_ - 1 && y > 0)
// 	    neighbor_pixels.push_back(*border_pixel_it + 1 - IMAGE_WIDTH_);
// 	  if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
// 	    neighbor_pixels.push_back(*border_pixel_it + 1 + IMAGE_WIDTH_);
// 	  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
// 	    int neighbor_segment_id = layer_surface_ids[*neighbor_pixel_it];
// 	    if (neighbor_segment_id == segment_id) {
// 	      if (visited_pixel_mask[*neighbor_pixel_it] == false) {
// 		new_border_pixels.push_back(*neighbor_pixel_it);
// 		visited_pixel_mask[*neighbor_pixel_it] = true;
// 	      }
// 	    } else {
// 	      neighbor_segment_counter[neighbor_segment_id]++;
// 	      boundary_length++;
// 	    }
// 	  }
// 	  boundary_length += 8 - neighbor_pixels.size();
// 	}
// 	if (new_border_pixels.size() == 0)
// 	  break;
// 	border_pixels = new_border_pixels;
//       }
//       // if (segment_id != 6)
//       // 	continue;

//       const int COMMON_BOUNDARY_LENGTH_THRESHOLD = boundary_length * COMMON_BOUNDARY_LENGTH_THRESHOLD_RATIO + 0.5;
//       set<int> neighbor_segment_ids;
//       for (map<int, int>::const_iterator neighbor_segment_it = neighbor_segment_counter.begin(); neighbor_segment_it != neighbor_segment_counter.end(); neighbor_segment_it++)
// 	if (neighbor_segment_it->second > COMMON_BOUNDARY_LENGTH_THRESHOLD)
// 	  neighbor_segment_ids.insert(neighbor_segment_it->first);
      
//       for (vector<int>::const_iterator segment_pixel_it = segment_pixels.begin(); segment_pixel_it != segment_pixels.end(); segment_pixel_it++)
//         for (set<int>::const_iterator neighbor_segment_it = neighbor_segment_ids.begin(); neighbor_segment_it != neighbor_segment_ids.end(); neighbor_segment_it++)
// 	  pixel_inpainting_surface_ids[*segment_pixel_it].insert(*neighbor_segment_it);

//       // if (segment_id == 1)
//       // 	for (vector<int>::const_iterator segment_pixel_it = segment_pixels.begin(); segment_pixel_it != segment_pixels.end(); segment_pixel_it++)
//       // 	  pixel_inpainting_surface_ids[*segment_pixel_it].push_back(4);

      
//       // if (neighbor_segments.count(current_solution_num_surfaces_) > 0) {
//       // 	assert(layer_index != NUM_LAYERS_ - 1);
//       // 	for (vector<int>::const_iterator segment_pixel_it = segment_pixels.begin(); segment_pixel_it != segment_pixels.end(); segment_pixel_it++)
//       // 	  pixel_inpainting_surface_ids[*segment_pixel_it] = current_solution_num_surfaces_;
//       // } else {
//       // 	for (vector<int>::const_iterator segment_pixel_it = segment_pixels.begin(); segment_pixel_it != segment_pixels.end(); segment_pixel_it++) {
//       // 	  double depth = current_solution_segments_[segment_id].getDepth(pixel);
//       // 	  double min_depth = 1000000;
//       // 	  int min_depth_segment_id = -1;
//       //     for (set<int>::const_iterator neighbor_segment_it = neighbor_segments.begin(); neighbor_segment_it != neighbor_segments.end(); neighbor_segment_it++) {
//       //       double neighbor_segment_depth = current_solution_segments_[*neighbor_segment_it].getDepth(pixel);
//       //       if (neighbor_segment_depth > depth - penalties_.depth_conflict_threshold && neighbor_segment_depth < min_depth) {
//       //         min_depth_segment_id = *neighbor_segment_it;
//       //         min_depth = neighbor_s
//       //       }
//       // 	  }
//       // 	}
//       // }
//     }
//     layer_pixel_inpainting_surface_ids[layer_index] = pixel_inpainting_surface_ids;
//   }

//   // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//   //   int current_solution_label = current_solution_labels_[pixel];
//   //   vector<bool> surface_fitting_mask(current_solution_num_surfaces_, false);
//   //   for (int surface_id = 0; surface_id < current_solution_num_surfaces_; surface_id++)
//   //     surface_fitting_mask[surface_id] = current_solution_segments_[surface_id].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel);
//   //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//   //     for (set<int>::const_iterator surface_it = layer_surface_ids_vec[layer_index].begin(); surface_it != layer_surface_ids_vec[layer_index].end(); surface_it++)
//   // 	if (surface_fitting_mask[*surface_it] == true)
//   // 	  layer_pixel_inpainting_surface_ids[layer_index][pixel].insert(*surface_it);
//   //     int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//   //     if (surface_id != current_solution_num_surfaces_)
//   // 	break;
//   //   }
//   // }

  
//   // vector<int> pixel_front_layer_index_map(NUM_PIXELS_);
//   // if (current_solution_num_surfaces_ > 0) {
//   //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//   //     int current_solution_label = current_solution_labels_[pixel];
//   //     int foremost_layer_index = NUM_LAYERS_ - 1;
//   //     for (int layer_index = NUM_LAYERS_ - 2; layer_index >= 0; layer_index--) {
//   //       int current_solution_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//   //       if (current_solution_surface_id < current_solution_num_surfaces_)
//   //         foremost_layer_index = layer_index;
//   //     }
//   //     pixel_foremost_layer_index_map[pixel] = foremost_layer_index;
//   //   }
//   // }
  
//   proposal_num_surfaces_ = current_solution_num_surfaces_;  
//   proposal_segments_ = current_solution_segments_;
  
//   proposal_labels_.assign(NUM_PIXELS_, vector<int>());
//   current_solution_indices_.assign(NUM_PIXELS_, 0);
//   int max_num_proposals = 0;  
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];
    
//     map<int, set<int> > pixel_layer_surfaces_map;
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       pixel_layer_surfaces_map[layer_index].insert(surface_id);
      
//       // bool is_visible = true;
//       // for (int target_layer_index = 0; target_layer_index <= layer_index - 1; target_layer_index++) {
//       // 	int target_layer_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - target_layer_index)) % (current_solution_num_surfaces_ + 1);
//       // 	if (target_layer_surface_id < current_solution_num_surfaces_)
//       // 	  is_visible = false;
//       // }
//       // //if (is_visible == true)
//       // for (int target_layer_index = 0; target_layer_index <= layer_index - 1; target_layer_index++)
//       //        pixel_layer_surfaces_map[target_layer_index].insert(surface_id);       

//       set<int> neighbor_segments = layer_pixel_inpainting_surface_ids[layer_index][pixel];
//       pixel_layer_surfaces_map[layer_index].insert(neighbor_segments.begin(), neighbor_segments.end());
//     }
//     for (int target_layer_index = 0; target_layer_index <= NUM_LAYERS_ - 1; target_layer_index++)
//       pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);
    
//     vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

//     vector<int> valid_pixel_proposals;
//     for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
//       if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
//         valid_pixel_proposals.push_back(*label_it);
    
//     // if (valid_pixel_proposals.size() > 1)
//     //   cout << "yes" << endl;
//     if (valid_pixel_proposals.size() == 0) {
//       cout << "empty proposal at pixel: " << pixel << endl;
//       // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
//       //   cout << pixel_proposal[proposal_index] << endl;
//       exit(1);
//     }      

//     proposal_labels_[pixel] = valid_pixel_proposals;
    
//     if (current_solution_num_surfaces_ > 0) {
//       current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
//       if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
//         cout << "has no current solution label at pixel: " << pixel << endl;
//         exit(1);
//       }
//     }
    
//     if (valid_pixel_proposals.size() > max_num_proposals) {
//       cout << "max number of proposals: " << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << valid_pixel_proposals.size() << endl;
//       max_num_proposals = valid_pixel_proposals.size();
//     }

//     // if (pixel == 47 * IMAGE_WIDTH_ + 184) {
//     //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
//     // 	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//     // 	  int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
//     // 	  cout << proposal_surface_id << '\t';
//     // 	}
//     // 	cout << endl;
//     //   }
//     //   exit(1);
//     // }
    

//     // pixel_proposals.push_back(current_solution_label);

//     // for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
//     //   int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//     //   if (surface_id != current_solution_num_surfaces_) {
//     // 	int proposal_label = current_solution_label + (current_solution_num_surfaces_ - surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index);
//     // 	pixel_proposals.push_back(proposal_label);
//     //   }
//     // }
//     // proposal_labels_[pixel] = pixel_proposals;
//   }

//   //addSegmentLayerProposals(true);
//   addIndicatorVariables();
  
  
//   // vector<int> correct_indicators(proposal_num_surfaces_ * NUM_LAYERS_ + NUM_LAYERS_, 0);
//   // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//   //   int label = current_solution_labels_[pixel];
//   //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//   //     int surface_id = label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//   //     if (surface_id < current_solution_num_surfaces_) {
//   //       correct_indicators[current_solution_num_surfaces_ * layer_index + surface_id] = 1;
//   //       correct_indicators[current_solution_num_surfaces_ * NUM_LAYERS_ + layer_index] = 1;
//   //     }
//   //   }
//   // }
//   // for (vector<int>::const_iterator indicator_it = correct_indicators.begin(); indicator_it != correct_indicators.end(); indicator_it++)
//   //   proposal_labels_.push_back(vector<int>(1, *indicator_it));

//   return true;
// }

bool ProposalDesigner::generateSegmentAddingProposal(const int denoted_segment_adding_type)
{
  cout << "generate segment adding proposal" << endl;
  proposal_type_ = "segment_adding_proposal";

  int segment_adding_type = denoted_segment_adding_type;
  if (segment_adding_type == -1)
    segment_adding_type = current_solution_num_surfaces_ == 0 ? 0 : rand() % 3 != 0 ? 1 : 1;
  segment_adding_type = current_solution_num_surfaces_ == 0 ? 0 : 1;
  
  //const double FITTING_DISTANCE_THRESHOLD = statistics_.fitting_distance_threshold;
  //const int SEGMENT_NUM_PIXELS_THRESHOLD = IMAGE_WIDTH_ * IMAGE_HEIGHT_ / 1000;
  
  vector<bool> bad_fitting_pixel_mask(NUM_PIXELS_, true);
  if (segment_adding_type != 0) {
    Mat bad_fitting_pixel_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int current_solution_label = current_solution_labels_[pixel];
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
	if (surface_id < current_solution_num_surfaces_) {
	  if (current_solution_segments_.at(surface_id).checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel) == false && (point_cloud_[pixel * 3 + 2] < current_solution_segments_.at(surface_id).getDepth(pixel) + statistics_.depth_conflict_threshold || layer_index == 0))
            bad_fitting_pixel_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = 255;
	  break;
	}
      }
    }
    Mat closed_bad_fitting_pixel_image = bad_fitting_pixel_image.clone();
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    for (int iteration = 0; iteration < 1; iteration++) {
      erode(closed_bad_fitting_pixel_image, closed_bad_fitting_pixel_image, element);
      dilate(closed_bad_fitting_pixel_image, closed_bad_fitting_pixel_image, element);
      erode(closed_bad_fitting_pixel_image, closed_bad_fitting_pixel_image, element);
      dilate(closed_bad_fitting_pixel_image, closed_bad_fitting_pixel_image, element);
    }
    
    imwrite("Test/bad_fitting_pixel_mask_image.bmp", closed_bad_fitting_pixel_image);
    //exit(1);
    bad_fitting_pixel_mask = vector<bool>(NUM_PIXELS_, false);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (closed_bad_fitting_pixel_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) > 128)
    	bad_fitting_pixel_mask[pixel] = true;
    
    //drawMaskImage(bad_fitting_pixel_mask, IMAGE_WIDTH_, IMAGE_HEIGHT_, "Test/bad_fitting_pixel_mask_image.bmp");
  }

  vector<double> visible_depths(NUM_PIXELS_, -1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        if (is_visible) {
          visible_depths[pixel] = depth;
          is_visible = false;
        }
      }
    }
  }

  
  // vector<vector<int> > new_segment_pixels_vec;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   if (bad_fitting_pixel_mask[pixel] == false)
  //     continue;
    
  //   vector<int> segment_pixels;
  //   vector<int> border_pixels;
  //   border_pixels.push_back(pixel);
  //   bad_fitting_pixel_mask[pixel] = false;
  //   while (true) {
  //     vector<int> new_border_pixels;
  //     for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
  // 	segment_pixels.push_back(*border_pixel_it);
  //       vector<int> neighbor_pixels;
  // 	int x = *border_pixel_it % IMAGE_WIDTH_;
  // 	int y = *border_pixel_it / IMAGE_WIDTH_;
  // 	if (x > 0)
  // 	  neighbor_pixels.push_back(*border_pixel_it - 1);
  // 	if (x < IMAGE_WIDTH_ - 1)
  // 	  neighbor_pixels.push_back(*border_pixel_it + 1);
  // 	if (y > 0)
  // 	  neighbor_pixels.push_back(*border_pixel_it - IMAGE_WIDTH_);
  // 	if (y < IMAGE_HEIGHT_ - 1)
  // 	  neighbor_pixels.push_back(*border_pixel_it + IMAGE_WIDTH_);
  // 	if (x > 0 && y > 0)
  // 	  neighbor_pixels.push_back(*border_pixel_it - 1 - IMAGE_WIDTH_);
  // 	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
  // 	  neighbor_pixels.push_back(*border_pixel_it - 1 + IMAGE_WIDTH_);
  // 	if (x < IMAGE_WIDTH_ - 1 && y > 0)
  // 	  neighbor_pixels.push_back(*border_pixel_it + 1 - IMAGE_WIDTH_);
  // 	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
  // 	  neighbor_pixels.push_back(*border_pixel_it + 1 + IMAGE_WIDTH_);
  // 	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
  // 	  if (bad_fitting_pixel_mask[*neighbor_pixel_it] == true) {
  // 	    new_border_pixels.push_back(*neighbor_pixel_it);
  // 	    bad_fitting_pixel_mask[*neighbor_pixel_it] = false;
  // 	  }
  // 	}
  //     }
  //     if (new_border_pixels.size() == 0)
  // 	break;
  //     border_pixels = new_border_pixels;
  //   }
  //   new_segment_pixels_vec.push_back(segment_pixels);
  // }  


  int foremost_empty_layer_index = NUM_LAYERS_ - 1;
  if (segment_adding_type == 1) {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int current_solution_label = current_solution_labels_[pixel];
      for (int layer_index = 0; layer_index < foremost_empty_layer_index + 1; layer_index++) {
	int current_solution_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
	if (current_solution_surface_id < current_solution_num_surfaces_)
	  foremost_empty_layer_index = layer_index - 1;
      }
      if (foremost_empty_layer_index == -1)
	break;
    }
  }
  
  proposal_segments_ = current_solution_segments_;
  vector<set<int> > pixel_segment_indices_map(NUM_PIXELS_);

  const int SMALL_SEGMENT_NUM_PIXELS_THRESHOLD = 10;
  int num_bad_fitting_pixels = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    if (bad_fitting_pixel_mask[pixel])
      num_bad_fitting_pixels++;
  const int NUM_FITTED_PIXELS_THRESHOLD = num_bad_fitting_pixels * 0.8;
  //  const int NUM_NEW_SEGMENTS_THRESHOLD = segment_adding_type == 0 ? 20 : segment_adding_type == 1 ? 5 : 1;
  
  {
    vector<bool> unfitted_pixel_mask = bad_fitting_pixel_mask;
    int num_fitted_pixels = 0;
    int proposal_segment_index = current_solution_num_surfaces_;
    //int new_proposal_segment_start_index = current_solution_num_surfaces_;
    //for (int proposal_segment_index = new_proposal_segment_start_index; proposal_segment_index < new_proposal_segment_start_index + NUM_NEW_SEGMENTS_THRESHOLD; proposal_segment_index++) {
    while (true) {
      vector<int> visible_pixels;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	if (unfitted_pixel_mask[pixel] == true)
	  visible_pixels.push_back(pixel);
      if (visible_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
        break;
      Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, visible_pixels, penalties_, statistics_);
      //vector<int> fitted_pixels = segment.getFittedPixels(point_cloud_, visible_pixels);
      vector<int> fitted_pixels = segment.getSegmentPixels();
      if (fitted_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
	break;
      if (segment.getType() < 0)
	continue;
      num_fitted_pixels += fitted_pixels.size();
      proposal_segments_[proposal_segment_index] = segment;
      for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++) {
	unfitted_pixel_mask[*pixel_it] = false;
	pixel_segment_indices_map[*pixel_it].insert(proposal_segment_index);
      }
      if (num_fitted_pixels > NUM_FITTED_PIXELS_THRESHOLD)
	break;
      proposal_segment_index++;
    }
  }

  {
    if (segment_adding_type != 0) {
      vector<bool> unfitted_pixel_mask = bad_fitting_pixel_mask;
      int proposal_segment_index = proposal_segments_.size();
      int num_new_planes = proposal_segments_.size() - current_solution_num_surfaces_;
      for (int i = 0; i < num_new_planes; i++) {
	vector<int> visible_pixels;
	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	  if (unfitted_pixel_mask[pixel] == true)
	    visible_pixels.push_back(pixel);
	if (visible_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
          break;

        Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, visible_pixels, penalties_, statistics_, 2);
      
	//vector<int> fitted_pixels = segment.getFittedPixels(point_cloud_, visible_pixels);
	vector<int> fitted_pixels = segment.getSegmentPixels();
	if (fitted_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
	  break;

	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	  unfitted_pixel_mask[*pixel_it] = false;

	if (fitted_pixels.size() > statistics_.bspline_surface_num_pixels_threshold || segment.getType() < 0)
	  continue;
      
	proposal_segments_[proposal_segment_index] = segment;
	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	  pixel_segment_indices_map[*pixel_it].insert(proposal_segment_index);
	proposal_segment_index++;
      }
    }
  }

  const int NUM_DILATION_ITERATIONS = 2;
  {
    //vector<bool> unfitted_pixel_mask = bad_fitting_pixel_mask;
    while (true) {
      bool has_change = false;
      
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        // if (pixel_segment_indices_map[pixel].size() == 0)
        //   continue;
        vector<int> neighbor_pixels;
        int x = pixel % IMAGE_WIDTH_;
        int y = pixel / IMAGE_WIDTH_;
        if (x > 0)
          neighbor_pixels.push_back(pixel - 1);
        if (x < IMAGE_WIDTH_ - 1)
          neighbor_pixels.push_back(pixel + 1);
        if (y > 0)
          neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
        if (y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
        if (x > 0 && y > 0)
          neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
        if (x > 0 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y > 0)
          neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
	  //   continue;
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (proposal_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it) && dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
	      dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	      has_change = true;
	    }
	  }
	}
      }
      if (has_change == false)
	break;
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
    }
    

    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        // if (pixel_segment_indices_map[pixel].size() == 0)
        //   continue;
        vector<int> neighbor_pixels;
        int x = pixel % IMAGE_WIDTH_;
        int y = pixel / IMAGE_WIDTH_;
        if (x > 0)
          neighbor_pixels.push_back(pixel - 1);
        if (x < IMAGE_WIDTH_ - 1)
          neighbor_pixels.push_back(pixel + 1);
        if (y > 0)
          neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
        if (y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
        if (x > 0 && y > 0)
          neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
        if (x > 0 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y > 0)
          neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
          // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
          //   continue;
          for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
              continue;
            if (proposal_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
              dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
          }
        }
      }
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
    }
  }


  if (segment_adding_type == 0) {
    const int NUM_PIXEL_SEGMENTS_THRESHOLD = 2;
    vector<bool> unfitted_pixel_mask = bad_fitting_pixel_mask;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (pixel_segment_indices_map[pixel].size() > 0)
        unfitted_pixel_mask[pixel] = true;
    while (true) {
      bool has_change = false;
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        if (unfitted_pixel_mask[pixel] == false)
          continue;
        vector<int> neighbor_pixels;
        int x = pixel % IMAGE_WIDTH_;
        int y = pixel / IMAGE_WIDTH_;
        if (x > 0)
          neighbor_pixels.push_back(pixel - 1);
        if (x < IMAGE_WIDTH_ - 1)
          neighbor_pixels.push_back(pixel + 1);
        if (y > 0)
          neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
        if (y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
        if (x > 0 && y > 0)
          neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
        if (x > 0 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y > 0)
          neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
          for (set<int>::const_iterator neighbor_segment_it = pixel_segment_indices_map[*neighbor_pixel_it].begin(); neighbor_segment_it != pixel_segment_indices_map[*neighbor_pixel_it].end(); neighbor_segment_it++) {
            if (proposal_segments_[*neighbor_segment_it].getDepth(pixel) > 0)
              new_pixel_segment_indices_map[pixel].insert(*neighbor_segment_it);
          }
        }
        
        if (new_pixel_segment_indices_map[pixel].size() >= NUM_PIXEL_SEGMENTS_THRESHOLD)
          unfitted_pixel_mask[pixel] = false;
        if (new_pixel_segment_indices_map[pixel].size() != pixel_segment_indices_map[pixel].size())
          has_change = true;
      }
      pixel_segment_indices_map = new_pixel_segment_indices_map;
      if (has_change == false)
        break;
    }
  }

  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, pixel_segment_indices_map);
  {
    Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
    map<int, Vec3b> color_table;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      if (layer_pixel_segment_indices_map[NUM_LAYERS_ - 1][pixel].size() == 0)
        continue;
      int segment_index = 1;
      for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++)
	if (*segment_it >= current_solution_num_surfaces_)
	  segment_index *= (*segment_it + 1);
      if (color_table.count(segment_index) == 0) {
        Vec3b color;
        for (int c = 0; c < 3; c++)
          color[c] = rand() % 256;
        color_table[segment_index] = color;
      }
      new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_index];
    }
    imwrite("Test/new_segment_image.bmp", new_segment_image);
  }
  
  // if (segment_adding_type == 2) {
  //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //     while (true) {
  //       bool has_change = false;
  //       vector<set<int> > expanded_pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
  //       for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //         // if (pixel_segment_indices_map[pixel].size() == 0)
  //         //   continue;
  //         vector<int> neighbor_pixels;
  //         int x = pixel % IMAGE_WIDTH_;
  //         int y = pixel / IMAGE_WIDTH_;
  //         if (x > 0)
  //           neighbor_pixels.push_back(pixel - 1);
  //         if (x < IMAGE_WIDTH_ - 1)
  //           neighbor_pixels.push_back(pixel + 1);
  //         if (y > 0)
  //           neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
  //         if (y < IMAGE_HEIGHT_ - 1)
  //           neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
  //         if (x > 0 && y > 0)
  //           neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
  //         if (x > 0 && y < IMAGE_HEIGHT_ - 1)
  //           neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
  //         if (x < IMAGE_WIDTH_ - 1 && y > 0)
  //           neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
  //         if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
  //           neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
  //         for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
  // 	    int current_solution_surface_id = current_solution_labels_[*neighbor_pixel_it] / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
  // 	    if (current_solution_surface_id == current_solution_num_surfaces_)
  // 	      continue;
  // 	    double current_solution_depth = current_solution_segments_.at(current_solution_surface_id).getDepth(*neighbor_pixel_it);
  // 	    for (set<int>::const_iterator segment_it = expanded_pixel_segment_indices_map[pixel].begin(); segment_it != expanded_pixel_segment_indices_map[pixel].end(); segment_it++) {
  // 	      double depth = proposal_segments_[*segment_it].getDepth(*neighbor_pixel_it);
  //             if (expanded_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0 && depth < current_solution_depth + statistics_.depth_conflict_threshold && depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_threshold) {
  //               expanded_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
  //               has_change = true;
  //             }
  //           }
  //         }
  //       }
  //       if (has_change == false)
  //         break;
  //       layer_pixel_segment_indices_map[layer_index] = expanded_pixel_segment_indices_map;
  //     }
  //   }
  // }

  // {
  //   Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //   map<int, Vec3b> color_table;
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     if (layer_pixel_segment_indices_map[NUM_LAYERS_ - 1][pixel].size() == 0)
  // 	continue;
  //     int segment_index = *layer_pixel_segment_indices_map[NUM_LAYERS_ - 1][pixel].begin();
  //     if (color_table.count(segment_index) == 0) {
  // 	Vec3b color;
  // 	for (int c = 0; c < 3; c++)
  // 	  color[c] = rand() % 256;
  // 	color_table[segment_index] = color;
  //     }
  //     new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_index];
  //   }
  //   imwrite("Test/new_segment_image.bmp", new_segment_image);
  //   exit(1);
  // }
      
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++)
  //     assert(proposal_segments_[*segment_it].getDepth(pixel) > 0);
  // cout << "done" << endl;
  
  cout << "number of new segments: " << proposal_segments_.size() - current_solution_num_surfaces_ << endl;
  if (proposal_segments_.size() - current_solution_num_surfaces_ == 0)
    return false;

  proposal_num_surfaces_ = proposal_segments_.size();
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  //int max_num_proposals = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];

    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_)
	pixel_layer_surfaces_map[layer_index].insert(surface_id);
      else
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }

    pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(pixel_segment_indices_map[pixel].begin(), pixel_segment_indices_map[pixel].end());
    if (segment_adding_type == 0 && pixel_segment_indices_map[pixel].size() == 0)
      for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++)
	pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(new_segment_index);

    // } else if (segment_adding_type == 1) {
    //   pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(pixel_segment_indices_map[pixel].begin(), pixel_segment_indices_map[pixel].end());
    // } else {
    //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    // 	pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
    // }

    
    // if (pixel_segment_index_map[pixel] != -1)
    //   //      for (int target_layer_index = 0; target_layer_index < NUM_LAYERS_; target_layer_index++)
    //   pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(pixel_segment_index_map[pixel]);
    // else
    //   for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++)
    // 	if (proposal_segments_[new_segment_index].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel) || bad_fitting_mask[pixel] == true)
    //       pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(new_segment_index);
      
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
	valid_pixel_proposals.push_back(*label_it);

    // if (valid_pixel_proposals.size() > 1)
    //   cout << "yes" << endl;
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
      //   cout << pixel_proposal[proposal_index] << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
	cout << "has no current solution label at pixel: " << pixel << endl;
	exit(1);
      }
    }
    
    
    //   // int new_segment_index = pixel_segment_index_map[pixel];
    //   // if (new_segment_index != -1) {
    //   //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //   // 	int surface_id = proposal_label / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //   // 	pixel_proposals.push_back(proposal_label + (new_segment_index - surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index));
    //   // 	if (surface_id < proposal_num_surfaces_)
    //   // 	  break;
    //   //   }
    //   // }

      
    // //   int foremost_layer_surface_id = proposal_label / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_  - 1 - max(foremost_empty_layer_index, 0))) % (proposal_num_surfaces_ + 1);
    // //   for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++) {
    // // 	if (pixel_segment_index_map[pixel] == -1 || new_segment_index == pixel_segment_index_map[pixel]) {
    // // 	  int new_proposal_label = proposal_label + (new_segment_index - foremost_layer_surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - max(foremost_empty_layer_index, 0));
    // // 	  if (checkLabelValidity(pixel, new_proposal_label, proposal_num_surfaces_, proposal_segments_))
    // //         pixel_proposals.push_back(new_proposal_label);
    // // 	}
    // //   }

    // //   if (foremost_empty_layer_index == -1 && foremost_layer_surface_id != proposal_num_surfaces_) {
    // // 	int new_proposal_label = proposal_label + (proposal_num_surfaces_ - foremost_layer_surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 0);
    // //     if (checkLabelValidity(pixel, new_proposal_label, proposal_num_surfaces_, proposal_segments_))
    // // 	  pixel_proposals.push_back(new_proposal_label);	
    // //   }
    // // } else {
    // //   for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++)
    // // 	if (pixel_segment_index_map[pixel] == -1 || new_segment_index == pixel_segment_index_map[pixel])
    // // 	  pixel_proposals.push_back(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_) - 1 - proposal_num_surfaces_ + new_segment_index);
    // // }
    
    // proposal_labels_[pixel] = pixel_proposals;
    // current_solution_indices_[pixel] = 0;  
      
    // if (pixel_proposals.size() > max_num_proposals) {
    //   cout << "max number of proposals: " << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << pixel_proposals.size() << endl;
    //   max_num_proposals = pixel_proposals.size();
    // }
  }
  //addSegmentLayerProposals(true);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateStructureExpansionProposal(const int denoted_expansion_layer_index, const int denoted_expansion_pixel)
{
  cout << "generate structure expansion proposal" << endl;
  proposal_type_ = "structure_expansion_proposal";

  vector<bool> candidate_segment_mask(current_solution_num_surfaces_, true);
  vector<double> visible_depths(NUM_PIXELS_, -1);
  vector<double> background_depths(NUM_PIXELS_, -1);
  vector<int> segment_backmost_layer_index_map(current_solution_num_surfaces_, 0);
  vector<int> visible_segmentation(NUM_PIXELS_, -1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    bool is_background = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        if (is_visible) {
	  visible_depths[pixel] = depth;
	  visible_segmentation[pixel] = surface_id;
          is_visible = false;
        }
	if (layer_index == NUM_LAYERS_ - 1) {
	  background_depths[pixel] = depth;
          candidate_segment_mask[surface_id] = false;
	}
	segment_backmost_layer_index_map[surface_id] = max(segment_backmost_layer_index_map[surface_id], layer_index);
      }
    }
  }
  for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
    if (segment_it->second.getType() != 0)
      candidate_segment_mask[segment_it->first] = false;
  }
  
  unique_ptr<StructureFinder> structure_finder(new StructureFinder(IMAGE_WIDTH_, IMAGE_HEIGHT_, current_solution_segments_, candidate_segment_mask, visible_segmentation, visible_depths, background_depths, segment_backmost_layer_index_map, penalties_, statistics_));
  vector<pair<double, vector<int> > > structure_score_surface_ids_pairs = structure_finder->getStructures();
  if (structure_score_surface_ids_pairs.size() == 0)
    return false;
  
  double score_sum = 0;
  for (int pair_index = 0; pair_index < structure_score_surface_ids_pairs.size(); pair_index++)
    score_sum += structure_score_surface_ids_pairs[pair_index].first;

  vector<int> structure_surface_ids;
  double selected_score = cv_utils::randomProbability() * score_sum;
  score_sum = 0;
  for (int pair_index = 0; pair_index < structure_score_surface_ids_pairs.size(); pair_index++) {
    score_sum += structure_score_surface_ids_pairs[pair_index].first;
    if (score_sum >= selected_score) {
      //cout << pair_index << endl;
      structure_surface_ids = structure_score_surface_ids_pairs[pair_index].second;
     break;
    }
  }

  int backmost_layer_index = 0;
  set<int> structure_surfaces;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int surface_id = structure_surface_ids[pixel];
    if (surface_id == -1)
      continue;
    backmost_layer_index = max(backmost_layer_index, segment_backmost_layer_index_map[surface_id]);
  }

  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);

      if (surface_id < proposal_num_surfaces_ && layer_index <= backmost_layer_index)
	for (int target_layer_index = 0; target_layer_index < layer_index; target_layer_index++)
	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);
    }
    if (structure_surface_ids[pixel] != -1)
      pixel_layer_surfaces_map[backmost_layer_index].insert(structure_surface_ids[pixel]);

    for (int target_layer_index = 0; target_layer_index < backmost_layer_index; target_layer_index++)
      pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);

    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }

  //addSegmentLayerProposals(false);
  addIndicatorVariables();

  return true;
}

bool ProposalDesigner::generateBackwardMergingProposal(const int denoted_target_layer_index)
{
  cout << "generate backward merging proposal" << endl;
  proposal_type_ = "backward_merging_proposal";
  
  int target_layer_index = denoted_target_layer_index;
  if (target_layer_index == -1) {
    int random_pixel = rand() % NUM_PIXELS_;
    int current_solution_label = current_solution_labels_[random_pixel];
    for (int layer_index = 1; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        target_layer_index = layer_index;
        break;
      }
    }
  }
  target_layer_index = NUM_LAYERS_ -1;

  vector<double> background_depths(NUM_PIXELS_, -1);
  vector<double> visible_depths(NUM_PIXELS_, -1);

  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    bool is_background = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
	if (is_visible) {
	  visible_depths[pixel] = depth;
	  is_visible = false;
	}
	if (layer_index >= target_layer_index && is_background) {
	  background_depths[pixel] = depth;
	  is_background = false;
	}
      }
    }
  }
  
  vector<set<int> > pixel_segment_indices_map(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < target_layer_index; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_ && current_solution_segments_[surface_id].getType() == 0)
	pixel_segment_indices_map[pixel].insert(surface_id);
    }
  }
  
  while (true) {
    bool has_change = false;
      
    vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      // if (pixel_segment_indices_map[pixel].size() == 0)
      //   continue;
      vector<int> neighbor_pixels;
      int x = pixel % IMAGE_WIDTH_;
      int y = pixel / IMAGE_WIDTH_;
      if (x > 0)
	neighbor_pixels.push_back(pixel - 1);
      if (x < IMAGE_WIDTH_ - 1)
	neighbor_pixels.push_back(pixel + 1);
      if (y > 0)
	neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
      if (y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
      if (x > 0 && y > 0)
	neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
      if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
      if (x < IMAGE_WIDTH_ - 1 && y > 0)
	neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
      if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	// if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
	//   continue;
	for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	  if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
	    continue;
	  double segment_depth = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it);
	  if (segment_depth < 0)
	    continue;
	  // if (*neighbor_pixel_it == 139 * IMAGE_WIDTH_ + 126) {
	  //   cout << segment_depth << '\t' << visible_depths[*neighbor_pixel_it] << endl;
	  //   exit(1);
	  // }
	  if ((segment_depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_threshold && segment_depth < background_depths[*neighbor_pixel_it] + statistics_.depth_conflict_threshold) || current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it)) {
	    new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	    has_change = true;
	  }
	}
      }
    }
    if (has_change == false)
      break;
    pixel_segment_indices_map = new_pixel_segment_indices_map;
  }
  
  const int NUM_DILATION_ITERATIONS = 2;
  for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
    vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      // if (pixel_segment_indices_map[pixel].size() == 0)
      //   continue;
      vector<int> neighbor_pixels;
      int x = pixel % IMAGE_WIDTH_;
      int y = pixel / IMAGE_WIDTH_;
      if (x > 0)
        neighbor_pixels.push_back(pixel - 1);
      if (x < IMAGE_WIDTH_ - 1)
        neighbor_pixels.push_back(pixel + 1);
      if (y > 0)
        neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
      if (y < IMAGE_HEIGHT_ - 1)
        neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
      if (x > 0 && y > 0)
        neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
      if (x > 0 && y < IMAGE_HEIGHT_ - 1)
        neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
      if (x < IMAGE_WIDTH_ - 1 && y > 0)
        neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
      if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
        neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
        // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
        //   continue;
	for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
          if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
            continue;
          if (current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
	    new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
        }
      }
    }
    pixel_segment_indices_map = new_pixel_segment_indices_map;
  }

  
  //  cout << pixel_segment_indices_map[139 * IMAGE_WIDTH_ + 126].size() << endl;
  Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  map<int, Vec3b> color_table;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    // if (pixel_segment_indices_map[pixel].size() == 0)
    //     continue;
    if (pixel_segment_indices_map[pixel].count(11) == 0)
      continue;
    int segment_index = 1;
    for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++)
      if (*segment_it >= current_solution_num_surfaces_)
        segment_index *= (*segment_it + 1);
    if (color_table.count(segment_index) == 0) {
      Vec3b color;
      for (int c = 0; c < 3; c++)
  	color[c] = rand() % 256;
      color_table[segment_index] = color;
    }
    new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = Vec3b(255, 255, 255);
    //new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_index];
  }
  imwrite("Test/backward_merging_image.bmp", new_segment_image);
  // exit(1);

  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      // if (surface_id != 2 && surface_id != 6)
      pixel_layer_surfaces_map[layer_index].insert(surface_id);

      if (layer_index < target_layer_index)
      	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }
    // if (pixel_surface_ids_map[pixel].count(2) > 0) {
    //   // if (pixel_layer_surfaces_map[0].count(2) == 0)
    //   // 	cout << pixel << endl;
    //   pixel_layer_surfaces_map[0].erase(2);
    //   pixel_layer_surfaces_map[0].insert(current_solution_num_surfaces_);
    //   pixel_layer_surfaces_map[target_layer_index].insert(2);
    // }
    pixel_layer_surfaces_map[target_layer_index].insert(pixel_segment_indices_map[pixel].begin(), pixel_segment_indices_map[pixel].end());

    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }

  //addSegmentLayerProposals(false);
  addIndicatorVariables();

  return true;
}

bool ProposalDesigner::generateBoundaryRefinementProposal()
{
  cout << "generate boundary refinement proposal" << endl;
  proposal_type_ = "boundary_refinement_proposal";
  
  vector<double> visible_depths(NUM_PIXELS_, -1);
  vector<double> background_depths(NUM_PIXELS_, -1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        if (is_visible) {
          visible_depths[pixel] = depth;
          is_visible = false;
        }
	if (layer_index == NUM_LAYERS_ - 1)
	  background_depths[pixel] = depth;
      }
    }
  }

  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_)
        layer_pixel_segment_indices_map[layer_index][pixel].insert(surface_id);
    }
  }

  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
    while (true) {
      bool has_change = false;
      
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	// if (pixel_segment_indices_map[pixel].size() == 0)
	//   continue;
	vector<int> neighbor_pixels;
	int x = pixel % IMAGE_WIDTH_;
	int y = pixel / IMAGE_WIDTH_;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
	  //   continue;
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	    if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
	      continue;
	    double segment_depth = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it);
	    if (segment_depth < 0)
	      continue;
	    // if (*neighbor_pixel_it == 139 * IMAGE_WIDTH_ + 126) {
	    //   cout << segment_depth << '\t' << visible_depths[*neighbor_pixel_it] << endl;
	    //   exit(1);
	    // }
	    if ((segment_depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_threshold && segment_depth < background_depths[*neighbor_pixel_it] + statistics_.depth_conflict_threshold) || current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it)) {
	      new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	      has_change = true;
	    }
	  
            // if (layer_index < NUM_LAYERS_ - 1) {
	    //   if (segment_depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_threshold || current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it)) {
	    // 	new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	    // 	has_change = true;
	    //   }
	    // } else {
	    //   if (segment_depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_threshold && segment_depth < background_depths[*neighbor_pixel_it]) {
            //     new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
            //     has_change = true;
            //   }
            // }
	  }
	}
      }
      if (has_change == false)
	break;
      pixel_segment_indices_map = new_pixel_segment_indices_map;
    }

    const int NUM_DILATION_ITERATIONS = 2;
    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	// if (pixel_segment_indices_map[pixel].size() == 0)
	//   continue;
	vector<int> neighbor_pixels;
	int x = pixel % IMAGE_WIDTH_;
	int y = pixel / IMAGE_WIDTH_;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
	  //   continue;
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	    if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
	      continue;
	    if (current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
	      new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	  }
	}
      }
      pixel_segment_indices_map = new_pixel_segment_indices_map;
    }
    layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  }


  // Mat boundary_refinement_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  // map<int, Vec3b> color_table;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   if (layer_pixel_segment_indices_map[2][pixel].count(3) > 0)
  //     boundary_refinement_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = 255;
  // imwrite("Test/boundary_refinement_image.bmp", boundary_refinement_image);
  // exit(1);
  
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    
    for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
      pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }

    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }


    // if (pixel == 2 * IMAGE_WIDTH_ + 108) {
    //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
    //  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //    int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //    cout << proposal_surface_id << '\t';
    //  }
    //  cout << endl;
    //   }
    //   exit(1);
    // }
  }

  //addSegmentLayerProposals(false);
  addIndicatorVariables();

  return true;
}

// bool ProposalDesigner::generateContourCompletionProposal()
// {
//   cout << "generate contour completion proposal" << endl;
//   proposal_type_ = "contour_completion_proposal";

//   vector<map<int, int> > segment_layer_votes(current_solution_num_surfaces_);
//   // map<int, map<int, int> > surface_depth_conflict_counter;
//   map<int, map<int, int> > segment_boundary_pixel_counter;
  
//   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//     vector<int> layer_surface_ids(NUM_PIXELS_);
//     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//       int current_solution_label = current_solution_labels_[pixel];
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       layer_surface_ids[pixel] = surface_id;
//     }

//     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//       int segment_id = layer_surface_ids[pixel];
//       if (segment_id == current_solution_num_surfaces_)
// 	continue;
//       vector<int> neighbor_pixels;
//       int x = pixel % IMAGE_WIDTH_;
//       int y = pixel / IMAGE_WIDTH_;
//       if (x > 0)
// 	neighbor_pixels.push_back(pixel - 1);
//       if (x < IMAGE_WIDTH_ - 1)
// 	neighbor_pixels.push_back(pixel + 1);
//       if (y > 0)
// 	neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
//       if (y < IMAGE_HEIGHT_ - 1)
// 	neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
//       if (x > 0 && y > 0)
// 	neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
//       if (x > 0 && y < IMAGE_HEIGHT_ - 1)
// 	neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
//       if (x < IMAGE_WIDTH_ - 1 && y > 0)
// 	neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
//       if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
// 	neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);

//       for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
// 	int neighbor_segment_id = layer_surface_ids[*neighbor_pixel_it];
// 	if (neighbor_segment_id == current_solution_num_surfaces_ || neighbor_segment_id == segment_id)
// 	  continue;
	
//         segment_boundary_pixel_counter[segment_id][neighbor_segment_id]++;
	
// 	double depth_1_1 = current_solution_segments_[segment_id].getDepth(pixel);
// 	double depth_1_2 = current_solution_segments_[segment_id].getDepth(*neighbor_pixel_it);
// 	double depth_2_1 = current_solution_segments_[neighbor_segment_id].getDepth(pixel);
// 	double depth_2_2 = current_solution_segments_[neighbor_segment_id].getDepth(*neighbor_pixel_it);
// 	if (depth_1_1 <= 0 || depth_1_2 <= 0 || depth_2_1 <= 0 || depth_2_2 <= 0)
// 	  continue;

//         double diff_1 = depth_1_1 - depth_2_1;
// 	double diff_2 = depth_1_2 - depth_2_2;
// 	if (diff_1 * diff_2 <= 0)
// 	  continue;

// 	if (max(diff_1, diff_2) < -statistics_.depth_change_smoothness_threshold) {
// 	  if (layer_index > 1)
// 	    segment_layer_votes[segment_id][layer_index - 1] += 1;
//           else
// 	    segment_layer_votes[neighbor_segment_id][layer_index + 1] += 1;
// 	}
//         if (min(diff_1, diff_2) > statistics_.depth_change_smoothness_threshold) {
//           if (layer_index > 1)
//             segment_layer_votes[neighbor_segment_id][layer_index - 1] += 1;
//           else
//             segment_layer_votes[segment_id][layer_index + 1] += 1;
//         }
//       }
//     }
//   }

//   vector<set<int> > layer_surface_ids_vec(NUM_LAYERS_);
//   vector<vector<int> > segment_visible_pixels_vec(current_solution_num_surfaces_);
//   vector<vector<int> > segment_pixels_vec(current_solution_num_surfaces_);
//   vector<bool> occluded_segment_mask(NUM_LAYERS_, false);
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];
//     bool is_visible = true;
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       if (surface_id < current_solution_num_surfaces_) {
// 	layer_surface_ids_vec[layer_index].insert(surface_id);
// 	if (is_visible == true) {
// 	  segment_visible_pixels_vec[surface_id].push_back(pixel);
//           is_visible = false;
// 	} else
//           occluded_segment_mask[surface_id] = true;
// 	segment_pixels_vec[surface_id].push_back(pixel);
//       }
//     }
//   }
  
//   map<int, vector<bool> > segment_completed_mask_map;
//   ContourCompleter contour_completer(IMAGE_WIDTH_, IMAGE_HEIGHT_);
//   for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
//     // if (occluded_segment_mask[segment_it->first] == true)
//     //   continue;
//     if (segment_it->first != 4)
//       continue;
//     vector<int> segment_pixels = segment_pixels_vec[segment_it->first];
//     vector<bool> segment_mask(NUM_PIXELS_, false);
//     for (vector<int>::const_iterator pixel_it = segment_pixels.begin(); pixel_it != segment_pixels.end(); pixel_it++)
//       segment_mask[*pixel_it] = true;
//     segment_completed_mask_map[segment_it->first] = contour_completer.completeContour(segment_mask, vector<bool>(NUM_PIXELS_, false));
//   }

//   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//     for (set<int>::const_iterator surface_it_1 = layer_surface_ids_vec[layer_index].begin(); surface_it_1 != layer_surface_ids_vec[layer_index].end(); surface_it_1++) {
//       if (segment_completed_mask_map.count(*surface_it_1) == 0)
//         continue;
//       for (set<int>::const_iterator surface_it_2 = layer_surface_ids_vec[layer_index].begin(); surface_it_2 != layer_surface_ids_vec[layer_index].end(); surface_it_2++) {
// 	if (*surface_it_1 >= *surface_it_2)
// 	  continue;
// 	if (segment_completed_mask_map.count(*surface_it_2) == 0)
//           continue;
// 	vector<bool> segment_mask_1 = segment_completed_mask_map[*surface_it_1];
// 	vector<bool> segment_mask_2 = segment_completed_mask_map[*surface_it_2];
// 	if (segment_boundary_pixel_counter.count(*surface_it_1) > 0 && segment_boundary_pixel_counter[*surface_it_1].count(*surface_it_2) > 0)
// 	  continue;
// 	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
// 	  if (segment_mask_1[pixel] == false || segment_mask_2[pixel] == false)
// 	    continue;
// 	  double depth_1 = current_solution_segments_[*surface_it_1].getDepth(pixel);
// 	  double depth_2 = current_solution_segments_[*surface_it_2].getDepth(pixel);

//           if (depth_1 <= 0 || depth_2 <= 0)
// 	    continue;

//           if (depth_1 < depth_2 - statistics_.depth_change_smoothness_threshold) {
// 	    if (layer_index > 1)
//               segment_layer_votes[*surface_it_1][layer_index - 1] += 1;
//             else
//               segment_layer_votes[*surface_it_2][layer_index + 1] += 1;
// 	  }
//           if (depth_2 < depth_1 - statistics_.depth_change_smoothness_threshold) {
//             if (layer_index > 1)
//               segment_layer_votes[*surface_it_2][layer_index - 1] += 1;
//             else
//               segment_layer_votes[*surface_it_1][layer_index + 1] += 1;
//           }
// 	}
//       }
//     }
//   }

//   vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];
//     bool is_visible = true;
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       if (surface_id < current_solution_num_surfaces_) {
// 	for (int target_layer_index = 0; target_layer_index <= NUM_LAYERS_; target_layer_index++)
// 	  if (target_layer_index != layer_index && segment_layer_votes[surface_id][target_layer_index] > 0)
// 	    layer_pixel_segment_indices_map[target_layer_index][pixel].insert(surface_id);
//       }
//     }
//   }
  
//   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//     for (set<int>::const_iterator surface_it = layer_surface_ids_vec[layer_index].begin(); surface_it != layer_surface_ids_vec[layer_index].end(); surface_it++) {
//       if (segment_completed_mask_map.count(*surface_it) == 0)
// 	continue;
//       vector<bool> segment_mask = segment_completed_mask_map[*surface_it];
//       for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
//         if (segment_mask[pixel] == true)
// 	  for (int target_layer_index = 0; target_layer_index <= NUM_LAYERS_; target_layer_index++)
//             if (target_layer_index == layer_index || segment_layer_votes[*surface_it][target_layer_index] > 0)
// 	      layer_pixel_segment_indices_map[target_layer_index][pixel].insert(*surface_it);
//     }
//   }

  
//   proposal_segments_ = current_solution_segments_;  
//   proposal_num_surfaces_ = current_solution_num_surfaces_;

//   proposal_labels_.assign(NUM_PIXELS_, vector<int>());
//   current_solution_indices_.assign(NUM_PIXELS_, 0);
//   //int max_num_proposals = 0;
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int current_solution_label = current_solution_labels_[pixel];
    
//     map<int, set<int> > pixel_layer_surfaces_map;
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
//       pixel_layer_surfaces_map[layer_index].insert(surface_id);
//     }

//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
//       pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());

//     for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++)
//       pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);

//     // for (set<int>::const_iterator surface_it = layer_surface_ids_vec[layer_index].begin(); surface_it != layer_surface_ids_vec[layer_index].end(); surface_it++)
//       //   if (segment_completed_mask_vec[*surface_it][pixel] == true)
//       // 	  pixel_layer_surfaces_map[layer_index].insert(*surface_it);
    
//     vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

//     vector<int> valid_pixel_proposals;
//     for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
//       if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
//         valid_pixel_proposals.push_back(*label_it);
    
//     if (valid_pixel_proposals.size() == 0) {
//       cout << "empty proposal at pixel: " << pixel << endl;
//       exit(1);
//     }

//     proposal_labels_[pixel] = valid_pixel_proposals;
    
//     if (current_solution_num_surfaces_ > 0) {
//       current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
//       if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
//         cout << "has no current solution label at pixel: " << pixel << endl;
//         exit(1);
//       }
//     }
    
//     // if (valid_pixel_proposals.size() > max_num_proposals) {
//     //   cout << "max number of proposals: " << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << valid_pixel_proposals.size() << endl;
//     //   max_num_proposals = valid_pixel_proposals.size();
//     // }
//   }

//   addIndicatorVariables();
// }

bool ProposalDesigner::generateDesiredProposal()
{
  cout << "generate desired proposal" << endl;
  proposal_type_ = "desired_proposal";
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  vector<int> visible_pixels;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    //proposal_labels_[pixel].push_back(current_solution_label);

    int layer_0_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 0)) % (current_solution_num_surfaces_ + 1);
    int layer_1_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 1)) % (current_solution_num_surfaces_ + 1);
    int layer_2_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 2)) % (current_solution_num_surfaces_ + 1);

    int proposal_label = current_solution_label;
    // if (layer_1_surface_id == 3) {
    //   if (current_solution_segments_[5].getDepth(pixel) < current_solution_segments_[4].getDepth(pixel))
    //   proposal_label -= 1;
    //   //visible_pixels.push_back(pixel);      
    // }
    
    if (layer_1_surface_id == 5 && layer_0_surface_id == 6)
      proposal_label += (current_solution_num_surfaces_ - layer_1_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 1);
    
    // if (current_solution_segments_[1].getDepth(pixel) < current_solution_segments_[layer_2_surface_id].getDepth(pixel))
    //   proposal_label += (1 - layer_2_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 2);
    
    // if (layer_0_surface_id == 4 || layer_0_surface_id == 7) {
    //   proposal_label += (current_solution_num_surfaces_ - layer_0_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 0) + (layer_0_surface_id - current_solution_num_surfaces_) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 1);
    // }
    
    proposal_labels_[pixel].push_back(proposal_label);
  }
  //  proposal_segments_[2].refitSegment(image_, point_cloud_, visible_pixels);

  //addSegmentLayerProposals(false);
  addIndicatorVariables();

  return true;
}

bool ProposalDesigner::generateSingleProposal()
{
  cout << "generate single proposal" << endl;
  proposal_type_ = "single_proposal";
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    proposal_labels_[pixel].push_back(current_solution_label);
  }
  //addSegmentLayerProposals(false);
  addIndicatorVariables();

  // Mat color_likelihood_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  // double max_color_likelihood = -1000000;
  // double min_color_likelihood = 0;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   color_likelihood_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min(max(proposal_segments_[7].predictColorLikelihood(pixel, blurred_hsv_image_.at<Vec3f>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_)) + 255, 0.0), 255.0);
  //   double color_likelihood = proposal_segments_[7].predictColorLikelihood(pixel, blurred_hsv_image_.at<Vec3f>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_));
  //   if (color_likelihood > max_color_likelihood)
  //     max_color_likelihood = color_likelihood;
  //   if (color_likelihood < min_color_likelihood)
  //     min_color_likelihood = color_likelihood; 
  // }
  // cout << max_color_likelihood << '\t' << min_color_likelihood << endl;
  // imwrite("Test/color_likelihood_image.bmp", color_likelihood_image);
  // exit(1);
  return true;
}

// bool ProposalDesigner::generateRansacProposal()
// {
//   cout << "generate ransac proposal" << endl;
//   proposal_type_ = "ransac_proposal";
  
//   proposal_num_surfaces_ = 3;
//   proposal_segments_.clear();
//   vector<bool> visible_pixel_mask(NUM_PIXELS_, true);
//   for (int segment_id = 0; segment_id < proposal_num_surfaces_; segment_id++) {
//     vector<int> visible_pixels;
//     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
//       if (visible_pixel_mask[pixel] == true)
// 	visible_pixels.push_back(pixel);
//     proposal_segments_[segment_id] = Segment(image_, point_cloud_, visible_pixels, vector<int>(visible_pixels.size(), -1), penalties_, statistics_);
//     vector<int> fitted_pixels = proposal_segments_[segment_id].getFittedPixels(point_cloud_, visible_pixels);
//     for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
//       visible_pixel_mask[*pixel_it] = false;
//   }
  
//   proposal_labels_.assign(NUM_PIXELS_, vector<int>());
//   current_solution_indices_.assign(NUM_PIXELS_, 0);
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     map<int, vector<int> > pixel_layer_surfaces_map;
//     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
//       for (int segment_id = 0; segment_id < proposal_num_surfaces_; segment_id++)
// 	pixel_layer_surfaces_map[layer_index].push_back(segment_id);
//       if (layer_index < NUM_LAYERS_ - 1)
//         pixel_layer_surfaces_map[layer_index].push_back(proposal_num_surfaces_);
//     }

//     vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

//     vector<int> valid_pixel_proposals;
//     for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
//       if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
//         valid_pixel_proposals.push_back(*label_it);

//     if (valid_pixel_proposals.size() == 0) {
//       cout << "empty proposal at pixel: " << pixel << endl;
//       exit(1);
//     }      

//     proposal_labels_[pixel] = valid_pixel_proposals;

//     if (current_solution_num_surfaces_ > 0) {
//       current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
//       if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
//         cout << "has no current solution label at pixel: " << pixel << endl;
//         exit(1);
//       }
//     }    
//   }
//   addIndicatorVariables();

//   return true;
// }

 void ProposalDesigner::initializeCurrentSolution()
{
  current_solution_labels_ = vector<int>(NUM_PIXELS_, 0);
  current_solution_num_surfaces_ = 0;
  current_solution_segments_.clear();
}

vector<int> ProposalDesigner::calcPixelProposals(const int num_surfaces, const map<int, set<int> > &pixel_layer_surfaces_map)
{
  vector<int> pixel_proposals(1, 0);
  for (map<int, set<int> >::const_iterator layer_it = pixel_layer_surfaces_map.begin(); layer_it != pixel_layer_surfaces_map.end(); layer_it++) {
    vector<int> new_pixel_proposals;
    for (set<int>::const_iterator segment_it = layer_it->second.begin(); segment_it != layer_it->second.end(); segment_it++)
      for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
	new_pixel_proposals.push_back(*label_it + *segment_it * pow(num_surfaces + 1, NUM_LAYERS_ - 1 - layer_it->first));
    pixel_proposals = new_pixel_proposals;
  }
  return pixel_proposals;
}

int ProposalDesigner::convertToProposalLabel(const int current_solution_label)
{
  int proposal_label = 0;
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
    if (surface_id == current_solution_num_surfaces_)
      proposal_label += proposal_num_surfaces_ * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index);
    else
      proposal_label += (surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index);
  }
  return proposal_label;
}

vector<int> ProposalDesigner::getCurrentSolutionIndices()
{
  return current_solution_indices_;
}

// vector<int> calcPixelProposals(const int num_layers, const int num_surfaces, const map<int, set<int> > &pixel_layer_surfaces_map)
// {
//   vector<int> pixel_proposals(1, 0);
//   for (map<int, set<int> >::const_iterator layer_it = pixel_layer_surfaces_map.begin(); layer_it != pixel_layer_surfaces_map.end(); layer_it++) {
//     vector<int> new_pixel_proposals;
//     for (set<int>::const_iterator segment_it = layer_it->second.begin(); segment_it != layer_it->second.end(); segment_it++)
//       for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
//         new_pixel_proposals.push_back(*label_it + *segment_it * pow(num_surfaces + 1, num_layers - 1 - layer_it->first));
//     pixel_proposals = new_pixel_proposals;
//   }
//   return pixel_proposals;
// }

void ProposalDesigner::getUpsamplingProposal(const Mat &ori_image, const vector<double> &ori_point_cloud, const vector<double> &ori_normals, const vector<double> &ori_camera_parameters, vector<vector<int> > &proposal_labels, int &proposal_num_surfaces, map<int, Segment> &proposal_segments, const int num_dilation_iterations)
{
  cout << "get upsampling proposal" << endl;

  const int ORI_IMAGE_WIDTH = ori_image.cols;
  const int ORI_IMAGE_HEIGHT = ori_image.rows;
  const int ORI_NUM_PIXELS = ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT;
  
  vector<vector<bool> > segment_masks(current_solution_num_surfaces_, vector<bool>(NUM_PIXELS_, false));
  vector<int> visible_segmentation(NUM_PIXELS_, -1);
  vector<set<int> > layer_surface_ids_vec(NUM_LAYERS_);
  vector<vector<bool> > layer_empty_masks(NUM_LAYERS_ - 1, vector<bool>(NUM_PIXELS_, false));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	segment_masks[surface_id][pixel] = true;
	layer_surface_ids_vec[layer_index].insert(surface_id);
	if (is_visible == true) {
	  visible_segmentation[pixel] = surface_id;
	  is_visible = false;
	}
      } else
	layer_empty_masks[layer_index][pixel] = true;
    }
  }
  
  for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++) {
    vector<bool> segment_mask = segment_masks[segment_id];
    for (int iteration = 0; iteration < num_dilation_iterations; iteration++) {
      vector<bool> dilated_segment_mask = segment_mask;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	if (segment_mask[pixel] == true)
	  continue;
	int x = pixel % IMAGE_WIDTH_;
	int y = pixel / IMAGE_WIDTH_;
	vector<int> neighbor_pixels;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (segment_mask[*neighbor_pixel_it]) {
	    dilated_segment_mask[pixel] = true;
	    break;
	  }
	}
      }
      segment_mask = dilated_segment_mask;
    }
    segment_masks[segment_id] = segment_mask;
  }

  for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
    vector<bool> empty_mask = layer_empty_masks[layer_index];
    for (int iteration = 0; iteration < num_dilation_iterations; iteration++) {
      vector<bool> dilated_empty_mask = empty_mask;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        if (empty_mask[pixel] == true)
          continue;
        int x = pixel % IMAGE_WIDTH_;
        int y = pixel / IMAGE_WIDTH_;
        vector<int> neighbor_pixels;
        if (x > 0)
          neighbor_pixels.push_back(pixel - 1);
        if (x < IMAGE_WIDTH_ - 1)
          neighbor_pixels.push_back(pixel + 1);
        if (y > 0)
          neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
        if (y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
        if (x > 0 && y > 0)
          neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
        if (x > 0 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y > 0)
          neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
        if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
          neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
          if (empty_mask[*neighbor_pixel_it]) {
            dilated_empty_mask[pixel] = true;
            break;
          }
        }
      }
      empty_mask = dilated_empty_mask;
    }
    layer_empty_masks[layer_index] = empty_mask;
  }
  
  vector<vector<int> > upsampled_segment_visible_pixels(current_solution_num_surfaces_);
  vector<vector<bool> > upsampled_segment_masks(current_solution_num_surfaces_, vector<bool>(ORI_NUM_PIXELS, false));
  vector<vector<bool> > upsampled_layer_empty_masks(NUM_LAYERS_ - 1, vector<bool>(ORI_NUM_PIXELS, false));
  for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
    int x = min(static_cast<int>(round(1.0 * (ori_pixel % ORI_IMAGE_WIDTH) / ORI_IMAGE_WIDTH * IMAGE_WIDTH_)), IMAGE_WIDTH_ - 1);
    int y = min(static_cast<int>(round(1.0 * (ori_pixel / ORI_IMAGE_WIDTH) / ORI_IMAGE_HEIGHT * IMAGE_HEIGHT_)), IMAGE_HEIGHT_ - 1);
    int pixel = y * IMAGE_WIDTH_ + x;
    upsampled_segment_visible_pixels[visible_segmentation[pixel]].push_back(ori_pixel);
    for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++)
      upsampled_segment_masks[segment_id][ori_pixel] = segment_masks[segment_id][pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++)
      upsampled_layer_empty_masks[layer_index][ori_pixel] = layer_empty_masks[layer_index][pixel];
  }

  
  proposal_segments.clear();
  for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++)
    proposal_segments[segment_id] = upsampleSegment(current_solution_segments_.at(segment_id), ori_image, ori_point_cloud, ori_normals, ori_camera_parameters, upsampled_segment_visible_pixels[segment_id]);
  proposal_num_surfaces = current_solution_num_surfaces_;
  
  proposal_labels.assign(ORI_NUM_PIXELS, vector<int>());
  for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      for (set<int>::const_iterator segment_it = layer_surface_ids_vec[layer_index].begin(); segment_it != layer_surface_ids_vec[layer_index].end(); segment_it++) {
	if (upsampled_segment_masks[*segment_it][ori_pixel])
	  pixel_layer_surfaces_map[layer_index].insert(*segment_it);
	if (layer_index < NUM_LAYERS_ - 1 && upsampled_layer_empty_masks[layer_index][ori_pixel])
          pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces);
      }
    }
    
    proposal_labels[ori_pixel] = calcPixelProposals(proposal_num_surfaces, pixel_layer_surfaces_map);
    // if (proposal_labels[ori_pixel].size() > 2)
    //   cout << ori_pixel << '\t' << proposal_labels[ori_pixel].size() << endl;
  }

  // vector<bool> layer_segment_indicators(proposal_num_surfaces * num_layers_, false);
  // for (int layer_index = 0; layer_index < num_layers_; layer_index++)
  //   for (set<int>::const_iterator segment_it = layer_surface_ids_vec[layer_index].begin(); segment_it != layer_surface_ids_vec[layer_index].end(); segment_it++)
  //     layer_segment_indicators[layer_index * proposal_num_surfaces + *segment_it] = true;
  // for (int i = 0; i < proposal_num_surfaces * num_layers_; i++)
  //   if (layer_segment_indicators[i] == true)
  //     proposal_labels.push_back(vector<int>(1, 1));
  //   else
  //     proposal_labels.push_back(vector<int>(1, 0));
}

void ProposalDesigner::writeSolution(const std::pair<double, LayerLabelSpace> &solution, const int thread_index, const int iteration) const
{
  time_t timer;
  time(&timer);
  struct tm today = {0};
  today.tm_hour = today.tm_min = today.tm_sec = 0;
  today.tm_mon = 2;
  today.tm_mday = 9;
  today.tm_year = 116;
  LOG(INFO) << difftime(timer, mktime(&today)) << '\t' << iteration << '\t' << thread_index << '\t' << solution.first << endl;
}
