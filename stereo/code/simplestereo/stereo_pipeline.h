//
// Created by yanhang on 3/17/16.
//

#ifndef PARALLELFUSION_STEREO_PIPELINE_H
#define PARALLELFUSION_STEREO_PIPELINE_H

#include "../../../base/LabelSpace.h"
#include "../../../base/ParallelFusionPipeline.h"

namespace simple_stereo {
class CompactLabelSpace : public ParallelFusion::LabelSpace<int> {
public:
  inline size_t getNumSingleLabel() const { return singleLabel.size(); }

  std::vector<int> &getSingleLabel() { return singleLabel; }

  const std::vector<int> &getSingleLabel() const { return singleLabel; }

  inline int operator[](const int id) const {
    CHECK_LT(id, singleLabel.size());
    return singleLabel[id];
  }

  const void appendSpace(const CompactLabelSpace &rhs) {
    CHECK(!(rhs.getLabelSpace().empty() && rhs.getSingleLabel().empty()));
    if (num_nodes_ == 0 && rhs.getNumNode() > 0) {
      label_space_.resize((size_t)rhs.getNumNode());
      num_nodes_ = (int)label_space_.size();
    }
    if (rhs.getNumNode() > 0) {
      for (auto i = 0; i < label_space_.size(); ++i) {
        for (auto j = 0; j < rhs.getLabelOfNode(i).size(); ++j)
          label_space_[i].push_back(rhs(i, j));
      }
    }
    if (rhs.getSingleLabel().size() > 0) {
      for (auto i = 0; i < rhs.getSingleLabel().size(); ++i)
        singleLabel.push_back(rhs.getSingleLabel()[i]);
    }
  }

  inline virtual bool empty() const {
    return singleLabel.empty() && label_space_.empty();
  }

  inline virtual void clear() {
    ParallelFusion::LabelSpace<int>::clear();
    singleLabel.clear();
  }

private:
  std::vector<int> singleLabel;
};

// customized pipeline. Simplify the logic in monitor thread
class StereoPipeline
    : public ParallelFusion::ParallelFusionPipeline<CompactLabelSpace> {
public:
  StereoPipeline(const ParallelFusion::ParallelFusionOption &option)
      : ParallelFusionPipeline(option) {}

  virtual void monitorThread(const int id, GeneratorPtr generator,
                             SolverPtr solver,
                             const ParallelFusion::ThreadOption &thread_option);
};

} // namespace simple_stereo
#endif // PARALLELFUSION_STEREO_PIPELINE_H
