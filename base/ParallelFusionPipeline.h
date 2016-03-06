#ifndef PARALLEL_FUSION_PIPELINE_H__
#define PARALLEL_FUSION_PIPELINE_H__

#include <memory>

#include "BaseDataStructures.h"
#include "ProposalGenerator.h"
#include "FusionSolver.h"
#include "FusionThread.h"

template<class LabelType> std::vector<LabelType> parallelFuse(std::vector<std::unique_ptr<FusionThread<LabelType> > > &fusion_threads, const PipelineParams &pipeline_params, const std::vector<LabelType> &initial_solution);

#endif
