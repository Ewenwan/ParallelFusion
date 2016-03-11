//
// Created by yanhang on 3/10/16.
//

#include "optimization.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace simple_stereo{
    double VictorOptimize::optimize(stereo_base::Depth &result, const int max_iter) const {
//        CompactLabelSpace all_solution;
//        parallelFusionPipeline.getAllResult(all_solution);
//        if(all_solution.getLabelSpace()[0].size() == 1)
//            parallelFusionPipeline.getBestLabeling(solution);
//        else{
//            printf("Performing final fusion...\n");
//            CompactLabelSpace fusedSolution;
//            fusedSolution.init(kPix, vector<int>(1,0));
//            for(auto i=0; i<kPix; ++i)
//                fusedSolution(i,0) = all_solution(i,0);
//            for(auto i=1; i<all_solution.getLabelSpace()[0].size(); ++i)
//                fuseTwoSolution(fusedSolution, all_solution, i, model);
//            solution.second =fusedSolution;
//            solution.first = solvers[0]->evaluateEnergy(solution.second);
//        }
    }
};

