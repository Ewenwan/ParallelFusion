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
#include "../../../base/LabelSpace.h"
#include "../../../base/FusionSolver.h"
#include "../../../base/ParallelFusionPipeline.h"
#include "../../../base/ProposalGenerator.h"

namespace simple_stereo {
    class SimpleStereo{
    public:
        SimpleStereo(const stereo_base::FileIO& file_io_, const int anchor_, const int dispResolution_);
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
        const int dispResolution;
        const double MRFRatio;
        std::vector<cv::Mat> images;
        std::vector<int> MRF_data;
        std::vector<int> hCue;
        std::vector<int> vCue;

        stereo_base::Depth unaryDisp;
        int width;
        int height;

    };

    class SimpleStereoSolver : public ParallelFusion::FusionSolver<int> {
    public:
        SimpleStereoSolver(){}
        virtual double solve(const ParallelFusion::LabelSpace<int> &proposals, std::vector<int> &solution) const;
        virtual double evaluateEnergy(const std::vector<int>& solution) const;
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
