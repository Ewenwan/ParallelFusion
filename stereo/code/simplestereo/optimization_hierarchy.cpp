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
            s = shared_ptr<FusionSolver<Space> >(new SimpleStereoSolver(model));

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

}//namespace simple_stereo

