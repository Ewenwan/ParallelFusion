//
// Created by yanhang on 3/3/16.
//

#ifndef SIMPLESTEREO_OPTIMIZATION_H
#define SIMPLESTEREO_OPTIMIZATION_H

#include <Eigen/Eigen>
#include <ctime>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "../stereo_base/depth.h"
#include "../stereo_base/file_io.h"

#include "../../../base/FusionSolver.h"
#include "../../../base/ParallelFusionPipeline.h"
#include "../../../base/ProposalGenerator.h"
#include "../external/MRF2.2/GCoptimization.h"
#include "stereo_pipeline.h"

namespace simple_stereo {
template <typename T> struct MRFModel {
  MRFModel()
      : MRF_data(NULL), hCue(NULL), vCue(NULL), width(0), height(0),
        MRFRatio(1000.0) {}
  ~MRFModel() { clear(); }

  int width;
  int height;
  int nLabel;
  T *MRF_data;
  T *hCue;
  T *vCue;
  T weight_smooth;
  double MRFRatio;

  inline void clear() {
    delete MRF_data;
    delete hCue;
    delete vCue;
    MRF_data = NULL;
    hCue = NULL;
    vCue = NULL;
    width = 0;
    height = 0;
    nLabel = 0;
  }

  inline T operator()(int pixId, int l) const {
    //            CHECK_LT(pixId, width * height);
    //            CHECK_LT(l, nLabel);
    return MRF_data[pixId * nLabel + l];
  }
  void init(const int w, const int h, const int n, const double wei) {
    MRF_data = new T[w * h * n];
    hCue = new T[w * h];
    vCue = new T[w * h];
    width = w;
    height = h;
    nLabel = n;
    weight_smooth = (T)(wei * MRFRatio);
  }

  inline T computeSmoothCost(const int pix, const int l1, const int l2,
                             bool direction) const {
    double cue = direction ? hCue[pix] : vCue[pix];
    return (T)((double)weight_smooth * (std::min(4, std::abs(l1 - l2))) * cue);
  }
};

class StereoOptimizer {
public:
  StereoOptimizer(const stereo_base::FileIO &file_io_,
                  const MRFModel<int> *model_, const std::string method_ = "")
      : file_io(file_io_), model(model_), width(model_->width),
        height(model_->height), nLabel(model_->nLabel), method(method_) {}
  virtual double optimize(stereo_base::Depth &result,
                          const int max_iter) const = 0;
  double evaluateEnergy(const std::vector<int> &labeling) const { return 0; }
  double evaluateEnergy(const stereo_base::Depth &result) const { return 0; }

protected:
  const stereo_base::FileIO &file_io;
  const MRFModel<int> *model;
  const int width;
  const int height;
  const int nLabel;
  const std::string method;
};

class FirstOrderOptimize : public StereoOptimizer {
public:
  FirstOrderOptimize(const stereo_base::FileIO &file_io_,
                     const MRFModel<int> *model_)
      : StereoOptimizer(file_io_, model_) {}
  virtual double optimize(stereo_base::Depth &result, const int max_iter) const;
};

class ParallelOptimize : public StereoOptimizer {
public:
  ParallelOptimize(const stereo_base::FileIO &file_io_,
                   const MRFModel<int> *model_, const int num_threads_,
                   const std::string method_,
                   const std::vector<int> &labelList_,
                   const bool multiway_ = false)
      : StereoOptimizer(file_io_, model_, method_), num_threads(num_threads_),
        labelList(labelList_), multiway(multiway_),
        num_proposals{4 + std::min(num_threads_ - 1, 1)}, exchange_interval{1},
        exchange_amount{std::min(num_threads_ - 1, 1)} {}

  ParallelOptimize(const stereo_base::FileIO &file_io_,
                   const MRFModel<int> *model_, const int num_threads_,
                   const std::string method_,
                   const std::vector<int> &labelList_, const int num_proposals_,
                   const int exchange_interval_, const int exchange_amount_,
                   const bool multiway_ = false)
      : StereoOptimizer(file_io_, model_, method_), num_threads(num_threads_),
        labelList(labelList_), num_proposals{num_proposals_},
        exchange_interval{exchange_interval_},
        exchange_amount{exchange_amount_}, multiway(multiway_) {}

  virtual double optimize(stereo_base::Depth &result, const int max_iter) const;
  void splitLabel(std::vector<std::vector<int>> &labelSubLists) const;

protected:
  const int num_threads, num_proposals, exchange_interval, exchange_amount;
  const std::vector<int> &labelList;
  const bool multiway;
};

class VictorOptimize : public ParallelOptimize {
public:
  VictorOptimize(const stereo_base::FileIO &file_io_,
                 const MRFModel<int> *model_, const int num_threads_,
                 const std::string method_, const std::vector<int> &labelList_,
                 const bool multiway_ = false)
      : ParallelOptimize(file_io_, model_, num_threads_, method_, labelList_,
                         multiway_) {}
  virtual double optimize(stereo_base::Depth &result, const int max_iter) const;
};

class HierarchyOptimize : public ParallelOptimize {
public:
  HierarchyOptimize(const stereo_base::FileIO &file_io_,
                    const MRFModel<int> *model_, const int num_threads_,
                    const std::vector<int> &labelList_)
      : ParallelOptimize(file_io_, model_, num_threads_, "Hierarchy",
                         labelList_, false) {}
  virtual double optimize(stereo_base::Depth &result, const int max_iter) const;
};

/////////////////////////////////////////////////////////////////////////
// Solvers
/////////////////////////////////////////////////////////////////////////
class SimpleStereoSolver
    : public ParallelFusion::FusionSolver<CompactLabelSpace> {
public:
  SimpleStereoSolver(const MRFModel<int> *model_)
      : model(model_), kPix(model->width * model->height) {}
  virtual void initSolver(const CompactLabelSpace &initial);
  virtual void
  solve(const CompactLabelSpace &proposals,
        const ParallelFusion::SolutionType<CompactLabelSpace> &current_solution,
        ParallelFusion::SolutionType<CompactLabelSpace> &solution);
  virtual double evaluateEnergy(const CompactLabelSpace &solution) const;

protected:
  const MRFModel<int> *model;
  const int kPix;
  std::shared_ptr<Expansion> mrf;
  std::shared_ptr<EnergyFunction> energy_function;
  std::shared_ptr<DataCost> dataCost;
  std::shared_ptr<SmoothnessCost> smoothnessCost;
};

class HierarchyStereoSolver : public SimpleStereoSolver {
public:
  HierarchyStereoSolver(const MRFModel<int> *model_)
      : SimpleStereoSolver(model_) {}
  virtual void
  solve(const CompactLabelSpace &proposals,
        const ParallelFusion::SolutionType<CompactLabelSpace> &current_solution,
        ParallelFusion::SolutionType<CompactLabelSpace> &solution);
};

class MultiwayStereoSolver : public SimpleStereoSolver {
public:
  MultiwayStereoSolver(const MRFModel<int> *model_)
      : SimpleStereoSolver(model_) {}
  virtual void
  solve(const CompactLabelSpace &proposals,
        const ParallelFusion::SolutionType<CompactLabelSpace> &current_solution,
        ParallelFusion::SolutionType<CompactLabelSpace> &solution);
};

class SimpleStereoMonitor : public SimpleStereoSolver {
public:
  SimpleStereoMonitor(const MRFModel<int> *model_)
      : SimpleStereoSolver(model_) {}
  virtual void initSolver(const CompactLabelSpace &initial);
  virtual void
  solve(const CompactLabelSpace &proposals,
        const ParallelFusion::SolutionType<CompactLabelSpace> &current_solution,
        ParallelFusion::SolutionType<CompactLabelSpace> &solution);
  void dumpData(const std::string &path) const;

private:
  std::vector<ParallelFusion::Observation> observations;
  std::vector<stereo_base::Depth> depths;
  std::vector<std::vector<stereo_base::Depth>> depths_threads;
  float start_time;
};

/////////////////////////////////////////////////////////////////////////
// Proposal Generators
/////////////////////////////////////////////////////////////////////////
class SimpleStereoGenerator
    : public ParallelFusion::ProposalGenerator<CompactLabelSpace> {
public:
  SimpleStereoGenerator(const int nPix_, const std::vector<int> &labelSubList_)
      : nPix(nPix_), labelTable(labelSubList_), nextLabel(0) {}
  virtual void getProposals(CompactLabelSpace &proposals,
                            const CompactLabelSpace &current_solution,
                            const int N);

protected:
  const int nPix;
  std::vector<int> labelTable;
  int nextLabel;
};

class MultiwayStereoGenerator : public SimpleStereoGenerator {
public:
  MultiwayStereoGenerator(const int nPix_,
                          const std::vector<int> &labelSubList_)
      : SimpleStereoGenerator(nPix_, labelSubList_) {}
  virtual void getProposals(CompactLabelSpace &proposals,
                            const CompactLabelSpace &current_solution,
                            const int N);
};

// generator for monitor thread, does nothing
class DummyGenerator
    : public ParallelFusion::ProposalGenerator<CompactLabelSpace> {
public:
  virtual void getProposals(CompactLabelSpace &proposals,
                            const CompactLabelSpace &current_solution,
                            const int N) {}
};

void fuseTwoSolution(CompactLabelSpace &s1, const CompactLabelSpace &s2,
                     const int pid, const MRFModel<int> *model);
void multiwayFusionByTRWS(const CompactLabelSpace &proposals,
                          const MRFModel<int> *model,
                          CompactLabelSpace &solution);
void dumpOutData(
    const ParallelFusion::ParallelFusionPipeline<CompactLabelSpace> &pipeline,
    const std::string &prefix);
} // namespace dynamic_stereo

#endif // DYNAMICSTEREO_OPTIMIZATION_H
