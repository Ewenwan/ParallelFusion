//
// Created by yanhang on 3/3/16.
//

#ifndef DYNAMICSTEREO_OPTIMIZATION_H
#define DYNAMICSTEREO_OPTIMIZATION_H

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "../stereo_base/depth.h"
#include "../stereo_base/file_io.h"

namespace sce_stereo {
    class StereoOptimization {
    public:
        typedef int EnergyType;

        StereoOptimization(const stereo_base::FileIO &file_io_, const int kFrames_, const cv::Mat &image_,
                           const std::vector<EnergyType> &MRF_data_, const float MRFRatio_, const int nLabel_, const int refId_) :
                file_io(file_io_), kFrames(kFrames_), image(image_), MRF_data(MRF_data_), MRFRatio(MRFRatio_),
                nLabel(nLabel_), refId(refId_), width(image_.cols), height(image_.rows) { }

        virtual void optimize(stereo_base::Depth &result, const int max_iter) const = 0;

        virtual double evaluateEnergy(const stereo_base::Depth &) const = 0;

        //vis = 255 means visible; vis = 0 means occluded
        void computeVisibility(const stereo_base::Depth& d, stereo_base::Depth& vis) const{
            vis.initialize(width, height, -1.0);
            std::vector<stereo_base::Depth> zBuffer((size_t)kFrames);
            for(auto& zb: zBuffer)
                zb.initialize(width, height, -1.0);
            for(auto v=0; v<kFrames; ++v) {
                for (auto x = 0; x < width; ++x) {
                    for (auto y = 0; y < height; ++y) {
                        double distance = (double) (v - (refId)) * d(x,y) / (double)nLabel * 64.0;
                        int tgtX = x + (int)round(distance+0.5);
                        if(tgtX >= 0 && tgtX < width)
                            zBuffer[v].setDepthAtInt(tgtX, y,std::max(zBuffer[v](tgtX, y), d(x,y)));
                    }
                }
            }

            for(auto x=0; x < width; ++x){
                for(auto y=0; y< height; ++y){
                    bool is_occl = false;
                    for(auto v=0; v<kFrames; ++v){
                        double distance = (double) (v - (refId)) * d(x,y) / (double)nLabel * 64.0;
                        int tgtX = x + (int)round(distance+0.5);
                        if(tgtX >= 0 && tgtX < width){
                            if(zBuffer[v](tgtX, y) > d(x,y)){
                                is_occl = true;
                                break;
                            }
                        }
                    }
                    if(is_occl)
                        vis.setDepthAtInt(x,y,0);
                    else
                        vis.setDepthAtInt(x,y,255.0);
                }
            }
        }

    protected:
        const stereo_base::FileIO &file_io;
        const int kFrames;
        const cv::Mat &image;
        const std::vector<EnergyType> &MRF_data;
        const int nLabel;
        const int refId;
        const float MRFRatio;

        const int width;
        const int height;
    };

    class FirstOrderOptimize : public StereoOptimization {
    public:
        FirstOrderOptimize(const stereo_base::FileIO &file_io_, const int kFrames_, const cv::Mat &image_,
                           const std::vector<EnergyType> &MRF_data_, const float MRFRatio_, const int nLabel_, const int refId_,
                           const EnergyType &weight_smooth_);

        virtual void optimize(stereo_base::Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const stereo_base::Depth &) const;

    private:
        void assignSmoothWeight();

        const EnergyType weight_smooth;
        std::vector<EnergyType> hCue;
        std::vector<EnergyType> vCue;
    };

    class SecondOrderOptimizeFusionMove : public StereoOptimization {
    public:
        SecondOrderOptimizeFusionMove(const stereo_base::FileIO &file_io_, const int kFrames_, const cv::Mat &image_,
                                      const std::vector<EnergyType> &MRF_data_,
                                      const float MRFRatio_,
                                      const int nLabel_, const int refId_,
                                      const stereo_base::Depth &noisyDisp_);

        virtual void optimize(stereo_base::Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const stereo_base::Depth &) const;

    private:
        void genProposal(std::vector<stereo_base::Depth> &proposals) const;

        void fusionMove(stereo_base::Depth &p1, const stereo_base::Depth &p2, const stereo_base::Depth& vis) const;
        inline double lapE(const double x0, const double x1, const double x2) const{
            return std::min(std::abs(x0 + x2 - 2 * x1), trun);
        }
        const stereo_base::Depth &noisyDisp;;
        const double trun;

        const int average_over;

        double laml;
        double lamh;
        std::vector<int> refSeg;
    };


}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_OPTIMIZATION_H
