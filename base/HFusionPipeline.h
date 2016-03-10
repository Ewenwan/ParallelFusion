//
// Created by yanhang on 3/9/16.
//

#ifndef PARALLELFUSION_HFUSIONPIPELINE_H
#define PARALLELFUSION_HFUSIONPIPELINE_H
#include "LabelSpace.h"
#include "ProposalGenerator.h"
#include "FusionSolver.h"

namespace ParallelFusion{

    struct HFusionPipelineOption{
    public:
        HFusionPipelineOption(): num_threads(4){}
        int num_threads;
    };

    template<class LABELSPACE>
    class HFusionPipeline{
    public:
        HFusionPipeline(const HFusionPipelineOption& option_): option(option_){}

    private:
        HFusionPipelineOption option;
    };
}//namespace ParallelFusion

#endif //PARALLELFUSION_HFUSIONPIPELINE_H