//
// Created by yanhang on 3/7/16.
//

#ifndef PARALLELFUSION_SIMPLESTEREO_H
#define PARALLELFUSION_SIMPLESTEREO_H

#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "../stereo_base/file_io.h"
#include "../stereo_base/depth.h"
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

    class SimpleStereo{
    public:
        SimpleStereo(const stereo_base::FileIO& file_io_, const int anchor_, const int dispResolution_, const double weight_smooth_);
        void initMRF();
        void computeMatchingCost();
        void assignSmoothWeight();

        void runStereo();

        inline int getWidth() const {return width;}
        inline int getHeight() const {return height;}

    private:
        typedef int EnergyType;

        const stereo_base::FileIO& file_io;
        const int anchor;
        std::vector<cv::Mat> images;

        MRFModel<int> model;

        stereo_base::Depth unaryDisp;
        int width;
        int height;

    };

    class SimpleStereoSolver : public ParallelFusion::FusionSolver<int> {
    public:
        SimpleStereoSolver(const MRFModel<int>& model_): model(model_){}
        virtual void initSolver(const std::vector<int>& initial);
        virtual double solve(const ParallelFusion::LabelSpace<int> &proposals, std::vector<int> &solution) const;
        virtual double evaluateEnergy(const std::vector<int>& solution) const;
    private:
        const MRFModel<int>& model;
    };

    class SimpleStereoGenerator: public ParallelFusion::ProposalGenerator<int>{
    public:
        SimpleStereoGenerator(const int nPix_, const int nLabel_, const int startid): nPix(nPix_), nLabel(nLabel_), nextLabel(startid % nLabel_){}
        virtual void getProposal(std::vector<int>& proposal, const std::vector<int>& current_solution);
    private:
        const int nPix;
        const int nLabel;
        int nextLabel;
    };
}

#endif //PARALLELFUSION_SIMPLESTEREO_H
