//
// Created by yanhang on 3/7/16.
//

#include "simplestereo.h"
#include "optimization.h"
#include "../stereo_base/local_matcher.h"

using namespace std;
using namespace stereo_base;
using namespace cv;
using namespace Eigen;
using namespace ParallelFusion;

namespace simple_stereo {

    SimpleStereo::SimpleStereo(const FileIO &file_io_, const int anchor_, const int dispResolution_,
                               const int downsample_, const double weight_smooth_, const int num_threads_) :
            file_io(file_io_), anchor(anchor_), downsample(downsample_), num_threads(num_threads_),
            model(new MRFModel<int>()) {
        CHECK_GE(file_io.getTotalNum(), 2) << "Too few images at " << file_io.getDirectory();
        images.resize((size_t) file_io.getTotalNum());
        CHECK(downsample == 1 || downsample == 2 || downsample == 4 || downsample == 8) << "Invalid downsample ratio!";
        const int nLevel = (int) std::log2((double) downsample) + 1;
        for (auto i = 0; i < images.size(); ++i) {
            vector<Mat> pyramid(nLevel);
            Mat tempMat = imread(file_io.getImage(i));
            pyramid[0] = tempMat;
            for (auto k = 1; k < nLevel; ++k)
                pyrDown(pyramid[k - 1], pyramid[k]);
            images[i] = pyramid.back().clone();
        }
        width = images[0].cols;
        height = images[0].rows;

        model->init(width, height, dispResolution_, weight_smooth_);
    }

    void SimpleStereo::initMRF() {
        printf("Computing matching cost\n");
        computeMatchingCost();
        printf("Assigning smoothness weight\n");
        assignSmoothWeight();
    }

    void SimpleStereo::computeMatchingCost() {
        //read from cache
        char buffer[1024] = {};
        sprintf(buffer, "%s/temp/cacheMRFdata", file_io.getDirectory().c_str());
        CHECK(model->MRF_data != NULL);

        ifstream fin(buffer, ios::binary);
        bool recompute = true;
        if (fin.is_open()) {
            int frame, resolution, type, ds;
            fin.read((char *) &frame, sizeof(int));
            fin.read((char *) &resolution, sizeof(int));
            fin.read((char *) &ds, sizeof(int));
            fin.read((char *) &type, sizeof(int));
            printf("Cached data: anchor:%d, resolution:%d, Energytype:%d\n",
                   frame, resolution, type);
            if (frame == anchor && resolution == model->nLabel && ds == downsample && type == sizeof(int)) {
                printf("Reading unary term from cache...\n");
                fin.read((char *) model->MRF_data, model->width * model->height * model->nLabel * sizeof(int));
                recompute = false;
            }
            fin.close();
        }
        if (recompute) {
            int index = 0;
            int unit = width * height / 10;
            //Be careful about down sample ratio!!!!!
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x, ++index) {
                    if (index % unit == 0)
                        cout << '.' << flush;
#pragma omp parallel for
                    for (int d = 0; d < model->nLabel; ++d) {
                        //project onto other views and compute matching cost
                        vector<vector<double>> patches(images.size());
                        for (auto v = 0; v < images.size(); ++v) {
                            double distance = (double) (v - anchor) * (double) d / downsample / 5;
                            Vector2d imgpt(x + distance, y);
                            local_matcher::samplePatch(images[v], imgpt, 3, patches[v]);
                        }
                        double mCost = local_matcher::sumMatchingCost(patches, anchor);
                        model->MRF_data[model->nLabel * (y * width + x) + d] = (int) ((mCost + 1) * model->MRFRatio);
                    }
                }
            }
            const int energyTypeSize = sizeof(EnergyType);
            ofstream cacheOut(buffer, ios::binary);
            cacheOut.write((char *) &anchor, sizeof(int));
            cacheOut.write((char *) &model->nLabel, sizeof(int));
            cacheOut.write((char *) &downsample, sizeof(int));
            cacheOut.write((char *) &energyTypeSize, sizeof(int));
            cacheOut.write((char *) model->MRF_data, sizeof(EnergyType) * model->width * model->height * model->nLabel);
            cacheOut.close();


        }
        cout << "Done" << endl;

        //compute unary disparity
        unaryDisp.initialize(width, height, -1);
        for (auto i = 0; i < width * height; ++i) {
            EnergyType min_e = std::numeric_limits<EnergyType>::max();
            for (auto d = 0; d < model->nLabel; ++d) {
                if (model->MRF_data[model->nLabel * i + d] < min_e) {
                    unaryDisp.setDepthAtInd(i, (double) d);
                    min_e = model->MRF_data[model->nLabel * i + d];
                }
            }
        }
    }

    void SimpleStereo::assignSmoothWeight() {
        const double t = 0.3;
        double ratio = 441.0;
        const Mat &img = images[anchor];
        for (auto y = 0; y < height; ++y) {
            for (auto x = 0; x < width; ++x) {
                Vec3b pix1 = img.at<Vec3b>(y, x);
                //pixel value range from 0 to 1, not 255!
                Vector3d dpix1 = Vector3d(pix1[0], pix1[1], pix1[2]) / 255.0;
                if (y < height - 1) {
                    Vec3b pix2 = img.at<Vec3b>(y + 1, x);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]) / 255.0;
                    double diff = (dpix1 - dpix2).norm();
                    if (diff > t)
                        model->vCue[y * width + x] = 0;
                    else
                        model->vCue[y * width + x] = (EnergyType) ((diff - t) * (diff - t) * ratio);
                }
                if (x < width - 1) {
                    Vec3b pix2 = img.at<Vec3b>(y, x + 1);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]) / 255.0;
                    double diff = (dpix1 - dpix2).norm();
                    if (diff > t)
                        model->hCue[y * width + x] = 0;
                    else
                        model->hCue[y * width + x] = (EnergyType) ((diff - t) * (diff - t) * ratio);
                }
            }
        }
    }


    void SimpleStereo::runStereo() {
        char buffer[1024] = {};
        initMRF();

        sprintf(buffer, "%s/temp/unaryDisp.jpg", file_io.getDirectory().c_str());
        unaryDisp.saveImage(buffer, 256.0 / (double) model->nLabel);

        vector<int> labelList((size_t)model->nLabel);
        for(auto i=0; i < labelList.size(); ++i)
            labelList[i] = i;
        random_shuffle(labelList.begin(), labelList.end());

        // cout << "========================================" << endl;
        // cout << "Runing sequential alpha-expansion" << endl;
        // ParallelOptimize optimize_sequential(file_io, model, 1, "Sequential", labelList);
        // Depth result_sequential;
        // optimize_sequential.optimize(result_sequential, 2);
        // sprintf(buffer, "%s/temp/result_sequential.jpg", file_io.getDirectory().c_str());
        // result_sequential.saveImage(buffer);

        // cout << "========================================" << endl;
        // cout << "Runing parallel method with solution sharing" << endl;
        // ParallelOptimize optimize_parallel(file_io, model, num_threads, "Swarn", labelList);
        // Depth result_parallel;
        // optimize_parallel.optimize(result_parallel, 2);
        // sprintf(buffer, "%s/temp/result_parallel.jpg", file_io.getDirectory().c_str());
        // result_parallel.saveImage(buffer);

        // cout << "========================================" << endl;
        // cout << "Runing multiway fusion with solution sharing" << endl;
        // ParallelOptimize optimize_multiway(file_io, model, num_threads, "Swarn_multiway", labelList, true);
        // Depth result_multiway;
        // optimize_multiway.optimize(result_multiway, 2);
        // sprintf(buffer, "%s/temp/result_multiway.jpg", file_io.getDirectory().c_str());
        // result_multiway.saveImage(buffer);

        cout << "========================================" << endl;
        cout << "Runing Victor's method" << endl;
        VictorOptimize optimize_victor(file_io, model, num_threads, "Victor", labelList);
        Depth result_victor;
        optimize_victor.optimize(result_victor, 1);
        sprintf(buffer, "%s/temp/result_victor.jpg", file_io.getDirectory().c_str());
        result_victor.saveImage(buffer);

        // cout << "========================================" << endl;
        // cout << "Runing Victor's method multiway" << endl;
        // VictorOptimize optimize_victor_multiway(file_io, model, num_threads, "Victor_multiway", labelList, true);
        // Depth result_victor_multiway;
        // optimize_victor_multiway.optimize(result_victor_multiway, 2);
        // sprintf(buffer, "%s/temp/result_victor_multiway.jpg", file_io.getDirectory().c_str());
        // result_victor_multiway.saveImage(buffer);

        // cout << "========================================" << endl;
        // cout << "Runing Hierarchy method" << endl;
        // HierarchyOptimize optimize_hierarchy(file_io, model, num_threads, labelList);
        // Depth result_hierarchy;
        // optimize_hierarchy.optimize(result_hierarchy, 1);
        // sprintf(buffer, "%s/temp/result_hierarchy.jpg", file_io.getDirectory().c_str());
        // cout << "Saving " << buffer << endl << flush;
        // result_hierarchy.saveImage(buffer);


//        FirstOrderOptimize optimize_firstOrder(file_io, model);
//        Depth result_firstOrder;
//        optimize_firstOrder.optimize(result_firstOrder, 1);
//        sprintf(buffer, "%s/temp/result_firstOrder.jpg", file_io.getDirectory().c_str());
//        result_firstOrder.saveImage(buffer);
    }
}//namespace simple_stereo
