#ifndef __LayerDepthMap__BinaryProposalDesigner__
#define __LayerDepthMap__BinaryProposalDesigner__

#include <vector>
#include <map>
#include <set>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <memory>

//#include "LayerInpainter.h"
//#include "GraphRepresenter.h"
//#include "LayerEstimator.h"
#include "Segment.h"
#include "LayerLabelSpace.h"

//using namespace cv;
//using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::Vector3d;


class BinaryProposalDesigner{

 public:
  BinaryProposalDesigner(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<double> &camera_parameters, const int num_layers, const RepresenterPenalties penalties, const DataStatistics statistics, const int scene_index);
  
  ~BinaryProposalDesigner();

  //generate a proposal
  bool getProposal(LayerLabelSpace &label_space, const int proposal_type_index);
  bool getProposal(int &iteration, std::vector<std::vector<int> > &proposal_labels, int &proposal_num_surfaces, std::map<int, Segment> &proposal_segments, std::string &proposal_type);
  bool getLastProposal(std::vector<std::vector<int> > &proposal_labels, int &proposal_num_surfaces, std::map<int, Segment> &proposal_segments, std::string &proposal_type);
  //std::string getProposalType();
  std::vector<int> getInitialLabels();
  void setCurrentSolution(const LayerLabelSpace &current_solution_label_space);
  void initializeCurrentSolution();
  std::vector<int> getCurrentSolutionIndices();
  void getUpsamplingProposal(const cv::Mat &ori_image, const std::vector<double> &ori_point_cloud, const std::vector<double> &ori_normals, const std::vector<double> &ori_camera_parameters, std::vector<std::vector<int> > &proposal_labels, int &proposal_num_surfaces, std::map<int, Segment> &proposal_segments, const int num_dilation_iterations);  
  bool getProposalAll(int &iteration, std::vector<std::vector<int> > &proposal_labels, int &proposal_num_surfaces, std::map<int, Segment> &proposal_segments, std::string &proposal_type);
  
 private:
  const cv::Mat image_;
  const std::vector<double> point_cloud_;
  const std::vector<double> normals_;
  const Eigen::MatrixXd projection_matrix_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  const std::vector<double> CAMERA_PARAMETERS_;
  const RepresenterPenalties penalties_;
  const DataStatistics statistics_;

  cv::Mat blurred_hsv_image_;
  
  std::vector<bool> ROI_mask_;
  int NUM_LAYERS_;
  const int NUM_PIXELS_;
  const int SCENE_INDEX_;

  std::vector<int> current_solution_labels_;
  int current_solution_num_surfaces_;
  std::map<int, Segment> current_solution_segments_;
  
  std::vector<std::vector<int> > proposal_labels_;
  int proposal_num_surfaces_;
  std::map<int, Segment> proposal_segments_;
  std::string proposal_type_;
  
  int num_confident_segments_threshold_;
  double segment_confidence_threshold_;


  std::vector<std::vector<int> > segmentations_;
  
  std::set<std::map<int, int> > used_confident_segment_layer_maps_;

  std::vector<int> current_solution_indices_;

  std::vector<int> single_surface_candidate_pixels_;

  std::vector<int> proposal_type_indices_;
  int proposal_type_index_ptr_;
  int all_proposal_iteration_;
  const int NUM_ALL_PROPOSAL_ITERATIONS_;
  
  
  bool generateSegmentationProposal(const int segmentation_index);
  bool generateSegmentRefittingProposal();
  bool generateSingleSurfaceExpansionProposal(const int segment_id = -1, const int denoted_expansion_type = -1);
  bool generateSurfaceDilationProposal();
  bool generateLayerSwapProposal();
  bool generateConcaveHullProposal(const bool consider_background);
  bool generateCleanUpProposal();
  bool generateSegmentDivisionProposal();
  bool generateSegmentAddingProposal(const int denoted_segment_adding_type = -1);
  bool generateStructureExpansionProposal(const int layer_index = -1, const int pixel = -1);
  bool generateBackwardMergingProposal(const int denoted_target_layer_index = -1);
  bool generateBoundaryRefinementProposal();
  bool generateBSplineSurfaceProposal();
  bool generateContourCompletionProposal();
  bool generateInpaintingProposal();
  bool generateDesiredProposal();
  bool generateSingleProposal();

  bool generateSegmentAddingProposalAll(const int denoted_segment_adding_type = 0);
  
  std::vector<int> calcPixelProposals(const int num_surfaces, const std::map<int, std::set<int> > &pixel_layer_surfaces_map);
  
  void convertProposalLabelsFormat();
  void addSegmentLayerProposals(const bool restrict_segment_in_one_layer);
  void addIndicatorVariables(const int num_indicator_variables = -1);
  
  bool checkLabelValidity(const int pixel, const int label, const int num_surfaces, const std::map<int, Segment> &segments);
  int convertToProposalLabel(const int current_solution_label);
  
  void writeSegmentationImage(const std::vector<int> &segmentation, const std::string filename);
  void writeSegmentationImage(const cv::Mat &segmentation_image, const std::string filename);

  void calcSegmentations();
};


#endif /* defined(__LayerDepthMap__BinaryProposalDesigner__) */
