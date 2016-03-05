#ifndef PARALLEL_FUSION_PIPELINE_H__
#define PARALLEL_FUSION_PIPELINE_H__

#include <memory>

#include "BaseDataStructures.h"
#include "ProposalGenerator.h"
#include "FusionSolver.h"
#include "FusionThread.h"

std::vector<int> parallelFuse(std::vector<std::unique_ptr<FusionThread> > &fusion_threads, const PipelineParams &pipeline_params, const std::vector<int> &initial_solution);

#endif
