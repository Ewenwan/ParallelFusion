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
#include "optimization.h"


namespace simple_stereo {



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


}

#endif //PARALLELFUSION_SIMPLESTEREO_H
