//
// Created by yanhang on 3/4/16.
//

#ifndef PARALLELFUSION_LOCAL_MATCHER_H
#define PARALLELFUSION_LOCAL_MATCHER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <Eigen/Eigen>
#include "../stereo_base/utility.h"
#include "../stereo_base/depth.h"
#include "../stereo_base/file_io.h"

namespace sce_stereo {
    namespace local_matcher {
        void samplePatch(const cv::Mat &img, const Eigen::Vector2d &loc, const int pR, std::vector<double> &pix);

        void getSSDArray(const std::vector<std::vector<double> > &patches, const int refId,
                         std::vector<double> &mCost);
        double sumMatchingCostHalf(const std::vector<std::vector<double> > &patches, const int refId);
    }
}
#endif //PARALLELFUSION_LOCAL_MATCHER_H
