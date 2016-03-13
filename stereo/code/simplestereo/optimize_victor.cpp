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

        vector<vector<int> > labelSubSpace;
        splitLabel(labelSubSpace);
        CHECK_EQ(labelSubSpace.size(), num_threads);

        //slave threads
        for(auto i=0; i<pipelineOption.num_threads; ++i){
            threadOptions[i].kTotal = kFusionSize;
            threadOptions[i].kOtherThread = 0;
            threadOptions[i].solution_exchange_interval = 1;
            initials[i].init(kPix, vector<int>(1, labelSubSpace[i].front()));
            if(multiway) {
                generators[i] = shared_ptr<ProposalGenerator<Space> >(new MultiwayStereoGenerator(kPix, labelSubSpace[i]));
                solvers[i] = shared_ptr<FusionSolver<Space> >(new MultiwayStereoSolver(model));
            }else{
                generators[i] = shared_ptr<ProposalGenerator<Space> >(new SimpleStereoGenerator(kPix, labelSubSpace[i]));
                solvers[i] = shared_ptr<FusionSolver<Space> >(new SimpleStereoSolver(model));
            }
            printf("Initial energy on thread %d: %.5f\n", i, solvers[i]->evaluateEnergy(initials[i]));
        }

        Pipeline victorFusionPipeline(pipelineOption);
        float start_t = (float)getTickCount();

        SolutionType<Space> solution;
        GlobalTimeEnergyProfile& profile =victorFusionPipeline.getGlobalProfile();
        for(auto iter=0; iter < max_iter; ++iter) {
            if(iter > 0){
                for(auto i=0; i<initials.size(); ++i)
                    initials[i] = solution.second;
            }
            bool reset_time = (iter == 0);
            victorFusionPipeline.runParallelFusion(initials, generators, solvers, threadOptions, reset_time);

            CompactLabelSpace all_solution;
            victorFusionPipeline.getAllResult(all_solution);
            if(all_solution.getLabelSpace()[0].size() == 1)
                victorFusionPipeline.getBestLabeling(solution);
            else{
                printf("Performing final fusion...\n");
                CompactLabelSpace fusedSolution;
                if(multiway){
                    multiwayFusionByTRWS(all_solution, model, solution.second);
                }else {

                    fusedSolution.init(kPix, vector<int>(1, 0));
                    for (auto i = 0; i < kPix; ++i)
                        fusedSolution(i, 0) = all_solution(i, 0);
                    for (auto i = 1; i < all_solution.getLabelSpace()[0].size(); ++i) {
                        cout << i << ' ' << flush;
                        fuseTwoSolution(fusedSolution, all_solution, i, model);
                    }
                    cout << endl;
                    solution.second = fusedSolution;
                }
                solution.first = solvers[0]->evaluateEnergy(solution.second);
            }

            float local_t = ((float)getTickCount() - start_t) / (float)getTickFrequency();
            profile.addObservation(local_t, solution.first);
        }
        float total_t = ((float)getTickCount() - start_t) / (float)getTickFrequency();
        printf("Done! Final energy: %.5f, running time: %.3fs\n", solution.first, total_t);

        dumpOutData(victorFusionPipeline, file_io.getDirectory()+"/temp/plot_"+method);

        for(auto i=0; i<model->width * model->height; ++i){
            result.setDepthAtInd(i, solution.second(i,0));
        }
        return solution.first;
    }
};

