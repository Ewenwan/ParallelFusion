//
// Created by yanhang on 3/17/16.
//

#include "stereo_pipeline.h"

using namespace ParallelFusion;
using namespace std;

namespace simple_stereo{

    void StereoPipeline::monitorThread(const int id, GeneratorPtr generator, SolverPtr solver, const ParallelFusion::ThreadOption &thread_option){
        try{
            printf("Monitor thread launched\n");
            solver->initSolver(CompactLabelSpace());
            while(true) {
                if(terminate.load()) {
                    printf("Monitor thread quited\n");
                    break;
                }
                std::this_thread::yield();

                CompactLabelSpace proposals;
                SolutionType<CompactLabelSpace> current_solution;

                current_solution.first = std::numeric_limits<double>::max();

                for (auto tid = 0; tid < slaveThreadIds.size(); ++tid) {
                    SolutionType<CompactLabelSpace> s;
                    bestSolutions[tid].get(s);
                    proposals.appendSpace(s.second);
                }

                SolutionType<CompactLabelSpace> curSolution;
                solver->solve(proposals, current_solution, curSolution);
                current_solution = curSolution;
            }
        }catch(const std::exception& e){
            terminate.store(true);
            printf("Thread %d throws and exception: %s\n", id, e.what());
            return;
        }
    }
}//namespace simple_stereo


