//
// Created by yanhang on 3/10/16.
//

#include "optimization.h"

using namespace std;
using namespace cv;
using namespace ParallelFusion;

namespace simple_stereo{
    double VictorOptimize::optimize(stereo_base::Depth &result, const int max_iter) const {
        typedef CompactLabelSpace Space;
        typedef ParallelFusionPipeline<Space> Pipeline;
        bool victorMethod = true;
        result.initialize(width, height, -1);
        const int kFusionSize = 4;
        ParallelFusionOption pipelineOption;
        pipelineOption.num_threads = num_threads;
        pipelineOption.max_iteration = model->nLabel / num_threads / kFusionSize;
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
            threadOptions[i].kTotal = kFusionSize;
            threadOptions[i].kOtherThread = 0;
            threadOptions[i].solution_exchange_interval = 1;
            printf("Thread %d, start: %d, interval:%d, num:%d\n", i, startid, pipelineOption.num_threads, kLabelPerThread);
            generators[i] = shared_ptr<ProposalGenerator<Space> >(new SimpleStereoGenerator(model->width * model->height, startid, interval, kLabelPerThread));
            solvers[i] = shared_ptr<FusionSolver<Space> >(new SimpleStereoSolver(model));
            printf("Initial energy on thread %d: %.5f\n", i, solvers[i]->evaluateEnergy(initials[i]));
        }

        Pipeline victorFusionPipeline(pipelineOption);
        float t = (float)getTickCount();
        victorFusionPipeline.runParallelFusion(initials, generators, solvers, threadOptions);

        SolutionType<Space> solution;
        CompactLabelSpace all_solution;
        victorFusionPipeline.getAllResult(all_solution);
        if(all_solution.getLabelSpace()[0].size() == 1)
            victorFusionPipeline.getBestLabeling(solution);
        else{
            printf("Performing final fusion...\n");
            CompactLabelSpace fusedSolution;
            fusedSolution.init(kPix, vector<int>(1,0));
            for(auto i=0; i<kPix; ++i)
                fusedSolution(i,0) = all_solution(i,0);
            for(auto i=1; i<all_solution.getLabelSpace()[0].size(); ++i)
                fuseTwoSolution(fusedSolution, all_solution, i, model);
            solution.second =fusedSolution;
            solution.first = solvers[0]->evaluateEnergy(solution.second);
        }

        t = ((float)getTickCount() - t) / (float)getTickFrequency();
        printf("Done! Final energy: %.5f, running time: %.3fs\n", solution.first, t);
        victorFusionPipeline.getGlobalProfile().addObservation(t, solution.first);

        dumpOutData(victorFusionPipeline, file_io.getDirectory()+"/temp/plot_victor");

        for(auto i=0; i<model->width * model->height; ++i){
            result.setDepthAtInd(i, solution.second(i,0));
        }
        return solution.first;
    }
};

