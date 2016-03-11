//
// Created by yanhang on 3/10/16.
//

#include "optimization.h"

using namespace std;
using namespace cv;
using namespace ParallelFusion;

namespace simple_stereo{
    double VictorOptimize::optimize(stereo_base::Depth &result, const int max_iter) const {\
        typedef CompactLabelSpace Space;
        typedef ParallelFusionPipeline<Space> Pipeline;

        bool victorMethod = true;
        result.initialize(width, height, -1);
        //configure as sequential fusion
        ParallelFusionOption pipelineOption;
        pipelineOption.num_threads = 4;
        pipelineOption.max_iteration = 16;
        const int kLabelPerThread = model->nLabel / pipelineOption.num_threads;

        Pipeline::GeneratorSet generators((size_t)pipelineOption.num_threads);
        Pipeline::SolverSet solvers((size_t)pipelineOption.num_threads);
        vector<Space> initials((size_t)pipelineOption.num_threads);
        vector<ThreadOption> threadOptions((size_t)pipelineOption.num_threads);

        const int kPix = model->width * model->height;

        //slave threads
        for(auto i=0; i<pipelineOption.num_threads; ++i){
            const int startid = i;
            const int interval = pipelineOption.num_threads;
            initials[i].init(kPix, vector<int>(1, startid));
            threadOptions[i].kTotal = 5;
            threadOptions[i].kOtherThread = 1;
            threadOptions[i].solution_exchange_interval = 1;
            printf("Thread %d, start: %d, interval:%d, num:%d\n", i, startid, pipelineOption.num_threads, kLabelPerThread);
            generators[i] = shared_ptr<ProposalGenerator<Space> >(new SimpleStereoGenerator(model->width * model->height, startid, interval, kLabelPerThread));
            solvers[i] = shared_ptr<FusionSolver<Space> >(new SimpleStereoSolver(model));
            printf("Initial energy on thread %d: %.5f\n", i, solvers[i]->evaluateEnergy(initials[i]));
        }

        Pipeline parallelFusionPipeline(pipelineOption);
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

