//
// Created by yanhang on 3/11/16.
//

#include "optimization.h"
using namespace cv;
using namespace std;
using namespace ParallelFusion;

namespace simple_stereo{
    void MultiwayStereoSolver::solve(const CompactLabelSpace &proposals,
                                     const ParallelFusion::SolutionType<CompactLabelSpace> &current_solution,
                                     ParallelFusion::SolutionType<CompactLabelSpace> &solution) {
        multiwayFusionByTRWS(proposals, model, solution.second);
        solution.first = evaluateEnergy(solution.second);
    }

    void MultiwayStereoGenerator::getProposals(CompactLabelSpace& proposals, const CompactLabelSpace& current_solution, const int N){
        proposals.appendSpace(current_solution);
        SimpleStereoGenerator::getProposals(proposals, current_solution, N-1);
    }

}//namespace simple_stereo

