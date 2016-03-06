//
// Created by yanhang on 3/4/16.
//

#ifndef PARALLELFUSION_SCESTEREO_H
#define PARALLELFUSION_SCESTEREO_H

#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "../stereo_base/file_io.h"
#include "../stereo_base/utility.h"
#include "../stereo_base/depth.h"

namespace sce_stereo {
    class SceStereo {
    public:
        SceStereo(const FileIO& file_io_, const int anchor_, const int tWindow_, const int resolution_);

        void runStereo();
    private:
        typedef int EnergyType;
        void initMRF();
        void computeMatchingCost();

        const FileIO& file_io;
        const int anchor;
        //const int tWindow;
        const int dispResolution;
        const double MRFRatio;
        const int pR;

        Depth unaryDisp;

        int offset;
        int width;
        int height;
        std::vector<cv::Mat> images;

        std::vector<EnergyType> MRF_data;
    };
}

#endif //PARALLELFUSION_SCESTEREO_H
