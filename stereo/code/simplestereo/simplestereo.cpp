//
// Created by yanhang on 3/7/16.
//

#include "simplestereo.h"

namespace simple_stereo {
    void SimpleStereoGenerator::getProposal(std::vector<int> &proposal,
                                            const std::vector<int> &current_solution) {

    }


    double SimpleStereoSolver::solve(const ParallelFusion::LabelSpace<int> &proposals,
                                     std::vector<int> &solution) const {
        return 0;
    }

    double SimpleStereoSolver::evaluateEnergy(const std::vector<int> &solution) const {
        return 0;
    }
}
