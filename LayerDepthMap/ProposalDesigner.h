#ifndef __LayerDepthMap__ProposalDesigner__
#define __LayerDepthMap__ProposalDesigner__

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>
#include <set>
#include <vector>

//#include "LayerInpainter.h"
//#include "GraphRepresenter.h"
//#include "LayerEstimator.h"
#include "Segment.h"

#include "../base/ProposalGenerator.h"
#include "BinaryProposalDesigner.h"
#include "LayerLabelSpace.h"

// using namespace cv;
// using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::Vector3d;

const std::string EDISON_PATH = "edison";
const std::string EDISON_EXE = "edison/edison edison/config.txt";

struct Solution {
  Solution() : thread_id(-1), time(-1), energy(-1), error(-1){};
  Solution(const int _thread_id, const int _time, const double _energy,
           const double _error, const std::vector<int> &_selected_threads,
           LayerLabelSpace _solution_space)
      : thread_id(_thread_id), time(_time), energy(_energy), error(_error),
        selected_threads(_selected_threads), solution_space(_solution_space){};

  int thread_id;
  int time;
  double energy;
  double error;
  std::vector<int> selected_threads;
  LayerLabelSpace solution_space;

  friend std::ostream &operator<<(std::ostream &out_str, Solution &solution) {
    out_str << solution.thread_id << '\t' << solution.time << '\t'
            << solution.energy << '\t' << solution.error << std::endl;
    out_str << solution.selected_threads.size() << std::endl;
    for (std::vector<int>::const_iterator thread_it =
             solution.selected_threads.begin();
         thread_it != solution.selected_threads.end(); thread_it++)
      out_str << *thread_it << '\t';
    out_str << std::endl;
    return out_str;
  }
  friend std::istream &operator>>(std::istream &in_str, Solution &solution) {
    in_str >> solution.thread_id >> solution.time >> solution.energy >>
        solution.error;
    int num_selected_threads = 0;
    in_str >> num_selected_threads;
    solution.selected_threads.assign(num_selected_threads, 0);
    for (int i = 0; i < num_selected_threads; i++)
      in_str >> solution.selected_threads[i];
    return in_str;
  }

  cv::Mat drawSolutionImage(const cv::Mat &image);
};

class ProposalDesigner
    : public ParallelFusion::ProposalGenerator<LayerLabelSpace> {

public:
  ProposalDesigner(const cv::Mat &image, const std::vector<double> &point_cloud,
                   const std::vector<double> &normals,
                   const std::vector<double> &camera_parameters,
                   const int num_layers, const RepresenterPenalties penalties,
                   const DataStatistics statistics, const int scene_index,
                   const bool use_concave_hull_proposal_first = true);

  ~ProposalDesigner();

  // generate a proposal
  void getProposals(LayerLabelSpace &proposal_label_space,
                    const LayerLabelSpace &current_solution, const int N);
  bool getLastProposal(std::vector<std::vector<int>> &proposal_labels,
                       int &proposal_num_surfaces,
                       std::map<int, Segment> &proposal_segments,
                       std::string &proposal_type);
  // std::string getProposalType();
  void getInitialLabelSpace(LayerLabelSpace &label_space);
  void setCurrentSolution(const LayerLabelSpace &current_solution_label_space);
  void initializeCurrentSolution();
  std::vector<int> getCurrentSolutionIndices();
  void getUpsamplingProposal(const cv::Mat &ori_image,
                             const std::vector<double> &ori_point_cloud,
                             const std::vector<double> &ori_normals,
                             const std::vector<double> &ori_camera_parameters,
                             std::vector<std::vector<int>> &proposal_labels,
                             int &proposal_num_surfaces,
                             std::map<int, Segment> &proposal_segments,
                             const int num_dilation_iterations);

  std::vector<Solution> getSolutions() { return solutions_; };
  void writeSolution(const std::pair<double, LayerLabelSpace> &solution,
                     const int thread_index, const int iteration,
                     const std::vector<int> &selected_threads);

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

  std::vector<std::vector<int>> proposal_labels_;
  int proposal_num_surfaces_;
  std::map<int, Segment> proposal_segments_;
  std::string proposal_type_;

  int num_confident_segments_threshold_;
  double segment_confidence_threshold_;

  std::vector<std::vector<int>> segmentations_;

  std::set<std::map<int, int>> used_confident_segment_layer_maps_;

  std::vector<int> current_solution_indices_;

  std::vector<int> single_surface_candidate_pixels_;

  std::vector<int> proposal_type_indices_;
  int proposal_type_index_ptr_;
  int all_proposal_iteration_;
  const int NUM_ALL_PROPOSAL_ITERATIONS_;

  int proposal_iteration_;

  std::unique_ptr<BinaryProposalDesigner> binary_proposal_designer_;
  const bool use_concave_hull_proposal_first_;

  std::vector<Solution> solutions_;
  std::vector<LayerLabelSpace> proposal_spaces_;

  bool generateSegmentRefittingProposal();
  bool
  generateSingleSurfaceExpansionProposal(const int segment_id = -1,
                                         const int denoted_expansion_type = -1);
  bool generateSurfaceDilationProposal();
  bool generateLayerSwapProposal();
  bool generateConcaveHullProposal(const bool consider_background);
  bool generateCleanUpProposal();
  bool generateSegmentDivisionProposal();
  bool
  generateSegmentAddingProposal(const int denoted_segment_adding_type = -1);
  bool generateStructureExpansionProposal(const int layer_index = -1,
                                          const int pixel = -1);
  bool
  generateBackwardMergingProposal(const int denoted_target_layer_index = -1);
  bool generateBoundaryRefinementProposal();
  bool generateBSplineSurfaceProposal();
  bool generateContourCompletionProposal();
  bool generateInpaintingProposal();
  bool generateDesiredProposal();
  bool generateSingleProposal();

  std::vector<int> calcPixelProposals(
      const int num_surfaces,
      const std::map<int, std::set<int>> &pixel_layer_surfaces_map);

  void convertProposalLabelsFormat();
  void addSegmentLayerProposals(const bool restrict_segment_in_one_layer);
  void addIndicatorVariables(const int num_indicator_variables = -1);

  bool checkLabelValidity(const int pixel, const int label,
                          const int num_surfaces,
                          const std::map<int, Segment> &segments);
  int convertToProposalLabel(const int current_solution_label);

  void writeSegmentationImage(const std::vector<int> &segmentation,
                              const std::string filename);
  void writeSegmentationImage(const cv::Mat &segmentation_image,
                              const std::string filename);
};

// std::vector<int> calcPixelProposals(const int num_layers, const int
// num_surfaces, const std::map<int, std::set<int> > &pixel_layer_surfaces_map);

#endif /* defined(__LayerDepthMap__ProposalDesigner__) */
