#ifndef FUSION_THREAD_H__
#define FUSION_THREAD_H__

#include <memory>

#include "BaseDataStructure.h"
#include "ProposalGenerator.h"
#include "FusionSolver.h"

void parallelFuse(const shared_ptr<ProposalGenerator> &proposal_generator, const shared_ptr<FusionSolver> &fusion_solver, const PipelineParams &pipeline_params);

#endif
