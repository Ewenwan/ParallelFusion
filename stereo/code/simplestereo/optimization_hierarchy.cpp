//
// Created by yanhang on 3/10/16.
//

#include "optimization.h"
#include "../../../base/HFusionPipeline.h"
using namespace std;
using namespace cv;
using namespace ParallelFusion;

namespace simple_stereo{

    double HierarchyOptimize::optimize(stereo_base::Depth &result, const int max_iter) const {
        typedef CompactLabelSpace Space;
        HFusionPipelineOption option;
        option.num_threads = num_threads;
        HFusionPipeline<Space> pipeline(option);

        std::vector<std::shared_ptr<FusionSolver<Space> > >solvers((size_t)option.num_threads);
        for(auto& s: solvers)
            s = shared_ptr<FusionSolver<Space> >(new HierarchyStereoSolver(model));

        std::vector<Space> proposals((size_t)model->nLabel);
        for(auto i=0; i<proposals.size(); ++i)
            proposals[i].getSingleLabel().push_back(i);

        Space initial;
        initial.init(width * height, vector<int>(1,0));

        float t = (float)getTickCount();
        pipeline.runHFusion(proposals, solvers, initial);
        t = ((float)getTickCount() - t) / (float)getTickFrequency();
        printf("Done. Time usage: %.3fs\n", t);
    }


    void HierarchyStereoSolver::solve(const CompactLabelSpace &proposals,
                                      const ParallelFusion::SolutionType<CompactLabelSpace> &current_solution,
                                      ParallelFusion::SolutionType<CompactLabelSpace> &solution) {
            solution.second.init(kPix, vector<int>(1,0));
            if(!proposals.getLabelSpace().empty()){
                    for(auto i=0; i<kPix; ++i)
                            solution.second(i,0) = proposals(i,0);
            }else{
                    CHECK_EQ(proposals.getSingleLabel().size(), 2);
                    for(auto i=0; i<kPix; ++i){
                            solution.second(i,0) = proposals.getSingleLabel().front();
                    }
            }
            if(!proposals.getSingleLabel().empty()){
                    //MRF
                    for(auto i=0; i<kPix; ++i)
                            mrf->setLabel(i, solution.second(i,0));
                    mrf->alpha_expansion(proposals.getSingleLabel().back());
                    for(auto i=0; i<kPix; ++i){
                            solution.second(i, 0) = mrf->getLabel(i);
                    }
            }else{
                    //QPBO
                    for(auto i=1; i<proposals.getLabelSpace()[0].size(); ++i)
                            fuseTwoSolution(solution.second, proposals, i, model);
            }
            solution.first = evaluateEnergy(solution.second);
            solution.second.getSingleLabel().clear();
    }
}//namespace simple_stereo

