//
// Created by yanhang on 3/3/16.
//

#ifndef SIMPLESTEREO_OPTIMIZATION_H
#define SIMPLESTEREO_OPTIMIZATION_H

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "../stereo_base/depth.h"
#include "../stereo_base/file_io.h"

#include "../external/MRF2.2/GCoptimization.h"
#include "../../../base/LabelSpace.h"
#include "../../../base/FusionSolver.h"
#include "../../../base/ParallelFusionPipeline.h"
#include "../../../base/ProposalGenerator.h"

namespace simple_stereo {
    template<typename T>
    struct MRFModel{
        MRFModel(): width(0), height(0), MRFRatio(100.0){}
        int width;
        int height;
        int nLabel;
        std::vector<T> MRF_data;
        std::vector<T> hCue;
        std::vector<T> vCue;
        T weight_smooth;
        const double MRFRatio;

        T operator()(int pixId, int l) const{
            CHECK_LT(pixId, width * height);
            CHECK_LT(l, nLabel);
            return MRF_data[pixId * nLabel + l];
        }
        void init(const int w, const int h, const int n, const double wei){
            MRF_data.resize((size_t)w * h * n);
            hCue.resize((size_t)w * h);
            vCue.resize((size_t)w * h);
            width = w;
            height = h;
            nLabel = n;
            weight_smooth = (T)(wei * MRFRatio);
        }
    };

    class StereoOptimizer{
    public:
        StereoOptimizer(const stereo_base::FileIO& file_io_, const MRFModel<int>& model_): file_io(file_io_), model(model_),
                                                                                      width(model_.width), height(model_.height), nLabel(model_.nLabel){}
        virtual double optimize(stereo_base::Depth& result, const int max_iter) const = 0;
        double evaluateEnergy(const std::vector<int>& labeling) const{
            return 0;
        }
        double evaluateEnergy(const stereo_base::Depth& result) const{
            return 0;
        }

    protected:
        const stereo_base::FileIO& file_io;
        const MRFModel<int>& model;
        const int width;
        const int height;
        const int nLabel;
    };

    class FirstOrderOptimize: public StereoOptimizer{
    public:
        FirstOrderOptimize(const stereo_base::FileIO &file_io_, const MRFModel<int>& model_): StereoOptimizer(file_io_, model_){}
        virtual double optimize(stereo_base::Depth &result, const int max_iter) const;
    };

    class ParallelOptimize: public StereoOptimizer{
    public:
        ParallelOptimize(const stereo_base::FileIO &file_io_, const MRFModel<int>& model_, const int num_threads_):
                StereoOptimizer(file_io_, model_), num_threads(num_threads_){}
        virtual double optimize(stereo_base::Depth &result, const int max_iter) const;
    private:
        const int num_threads;
    };


    class SimpleStereoSolver : public ParallelFusion::FusionSolver<ParallelFusion::LabelSpace<int> > {
    public:
        SimpleStereoSolver(const MRFModel<int>& model_): model(model_), kPix(model.width * model.height){}
        virtual void initSolver(const ParallelFusion::LabelSpace<int>& initial);
        virtual double solve(const ParallelFusion::LabelSpace<int> &proposals, ParallelFusion::LabelSpace<int> &solution) const;
        virtual double evaluateEnergy(const ParallelFusion::LabelSpace<int>& solution) const;
    private:
        inline int smoothnessCost(int pix, int l1, int l2, bool xDirection) const{
            double cue = xDirection ? model.hCue[pix] : model.vCue[pix];
            return (int)((double)model.weight_smooth * (std::min(4, std::abs(l1-l2))) * cue);
        }
        const MRFModel<int>& model;
        const int kPix;
        std::shared_ptr<Expansion> mrf;
    };

    class SimpleStereoGenerator: public ParallelFusion::ProposalGenerator<ParallelFusion::LabelSpace<int> >{
    public:
        SimpleStereoGenerator(const int nPix_, const int nLabel_, const int startid): nPix(nPix_), nLabel(nLabel_), nextLabel(startid % nLabel_){}
        virtual void getProposals(ParallelFusion::LabelSpace<int>& proposals, const ParallelFusion::LabelSpace<int>& current_solution, const int N);
    private:
        const int nPix;
        const int nLabel;
        int nextLabel;
    };


}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_OPTIMIZATION_H
