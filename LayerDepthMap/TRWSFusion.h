#ifndef __LayerDepthMap__TRWSFusion__
#define __LayerDepthMap__TRWSFusion__

#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <set>
#include <stdio.h>
#include <vector>

#include "DataStructure.h"
#include "Segment.h"
#include "TRW_S/MRFEnergy.h"

#include "../base/FusionSolver.h"
#include "LayerLabelSpace.h"

using namespace std;

class TRWSFusion : public ParallelFusion::FusionSolver<LayerLabelSpace> {
public:
  TRWSFusion(const cv::Mat &image, const vector<double> &point_cloud,
             const vector<double> &normals,
             const RepresenterPenalties &penalties,
             const DataStatistics &statistics,
             const bool consider_surface_cost = true);
  // Copy constructor
  // TRWSFusion(TRWSFusion &solver);

  // Destructor
  ~TRWSFusion();

  //    bool Save(char* filename);
  //    bool Load(char* filename);

  // void addUnaryTerm(const int i, const int label, const double E);
  // void addPairwiseTerm(const int i, const int j, const int label_i, const int
  // label_j, const double E);

  void
  solve(const LayerLabelSpace &proposal_label_space,
        const ParallelFusion::SolutionType<LayerLabelSpace> &current_solution,
        ParallelFusion::SolutionType<LayerLabelSpace> &solution);
  double evaluateEnergy(const LayerLabelSpace &solution) const {
    return 100000000;
  };

private:
  const int IMAGE_WIDTH_, IMAGE_HEIGHT_, NUM_PIXELS_;
  const cv::Mat image_;
  cv::Mat blurred_hsv_image_;
  const vector<double> point_cloud_;
  const vector<double> normals_;
  const RepresenterPenalties penalties_;
  const DataStatistics statistics_;
  const bool consider_surface_cost_;

  int proposal_num_segments_;
  int proposal_num_layers_;
  map<int, Segment> proposal_segments_;

  vector<double> boundary_scores_;

  double color_diff_var_;

  MRFEnergy<TypeGeneral> *initializeEnergyFromCalculation();
  MRFEnergy<TypeGeneral> *initializeEnergyFromFile();

  double calcDataCost(const int pixel, const int label);
  double calcSmoothnessCost(const int pixel_1, const int pixel_2,
                            const int label_1, const int label_2);
  double calcSmoothnessCostMulti(const int pixel_1, const int pixel_2,
                                 const int label_1, const int label_2);

  double checkSolutionEnergy(const vector<int> &solution_for_check);
  vector<int> getOptimalSolution();

  // void calcDepthSVar();
  void calcBoundaryScores();

  void calcColorDiffVar();
  double calcColorDiff(const int pixel_1, const int pixel_2);

  vector<vector<set<int>>>
  calcOverlapPixels(const vector<vector<int>> &proposal_labels);
};

#endif /* defined(__LayerDepthMap__TRWSFusion__) */
