//
// Created by yanhang on 3/7/16.
//

#ifndef PARALLELFUSION_SIMPLESTEREO_H
#define PARALLELFUSION_SIMPLESTEREO_H

#include <glog/logging.h>
#include <iostream>

#include "../../../base/LabelSpace.h"
#include "../../../base/FusionSolver.h"
#include "../../../base/ParallelFusionPipeline.h"
#include "../../../base/ProposalGenerator.h"

namespace simple_stereo {
    class SimpleStereoSolver : public ParallelFusion::FusionSolver<int> {
    public:
        virtual double solve(const ParallelFusion::LabelSpace<int> &proposals, std::vector<int> &solution) const;
        virtual double evaluateEnergy(const std::vector<int>& solution) const;
    private:

    };

    class SimpleStereoGenerator: public ParallelFusion::ProposalGenerator<int>{
    public:
        SimpleStereoGenerator(const int nPix_, const int nLabel_, const int startid): nPix(nPix_), nLabel(nLabel_), nextLabel(startid % nLabel_){}
        virtual void getProposal(std::vector<int>& proposal, const std::vector<int>& current_solution);
    private:
        const int nPix;
        const int nLabel;
        int nextLabel;
    };
}

#endif //PARALLELFUSION_SIMPLESTEREO_H
