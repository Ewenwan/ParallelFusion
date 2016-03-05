//
// Created by yanhang on 3/4/16.
//

#include "scestereo.h"
#include "local_matcher.h"
#include "proposal.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace sce_stereo{

    SceStereo::SceStereo(const FileIO &file_io_, const int anchor_, const int tWindow_, const int resolution_):
            file_io(file_io_), anchor(anchor_), tWindow(tWindow_), dispResolution(resolution_), MRFRatio(1000), pR(3){
        CHECK_GE(file_io.getTotalNum(), 2) << "Too few images";
        int tR = tWindow / 2;
        offset = std::max(anchor-tR, 0);
        CHECK_LE(offset + tWindow, file_io.getTotalNum());

        images.resize((size_t)tWindow);
        for(auto i=0; i<images.size(); ++i)
            images[i] = imread(file_io.getImage(i));
        width = images[0].cols;
        height = images[0].rows;
    }

    void SceStereo::initMRF() {
        MRF_data.resize((size_t)(width*height*dispResolution), (EnergyType)0);
        printf("Computing matching cost...\n");
        computeMatchingCost();
    }

    void SceStereo::computeMatchingCost() {
        //read from cache
        char buffer[1024] = {};
        sprintf(buffer, "%s/temp/cacheMRFdata", file_io.getDirectory().c_str());
        ifstream fin(buffer, ios::binary);

        bool recompute = true;
        if (fin.is_open()) {
            int frame, resolution, tw, type;
            fin.read((char *) &frame, sizeof(int));
            fin.read((char *) &resolution, sizeof(int));
            fin.read((char *) &tw, sizeof(int));
            fin.read((char *) &type, sizeof(int));
            printf("Cached data: anchor:%d, resolution:%d, twindow:%d, Energytype:%d\n",
                   frame, resolution, tw, type);
            if (frame == anchor && resolution == dispResolution && tw == tWindow &&
                type == sizeof(EnergyType)) {
                printf("Reading unary term from cache...\n");
                fin.read((char *) MRF_data.data(), MRF_data.size() * sizeof(EnergyType));
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
                    for (int d = 0; d < dispResolution; ++d) {
                        //project onto other views and compute matching cost
                        vector<vector<double>> patches(images.size());
                        for (auto v = 0; v < images.size(); ++v) {
                            double distance = (double) (v - (anchor - offset)) * (double)d;
                            Vector2d imgpt(x - distance, y);
                            local_matcher::samplePatch(images[v], imgpt, 3, patches[v]);
                        }
                        double mCost = local_matcher::sumMatchingCostHalf(patches, anchor - offset);
                        MRF_data[dispResolution * (y * width + x) + d] = (EnergyType) ((mCost + 1) * MRFRatio);
                    }
                }
            }

            const int energyTypeSize = sizeof(EnergyType);
            ofstream cacheOut(buffer, ios::binary);
            cacheOut.write((char *) &anchor, sizeof(int));
            cacheOut.write((char *) &dispResolution, sizeof(int));
            cacheOut.write((char *) &tWindow, sizeof(int));
            cacheOut.write((char *) &energyTypeSize, sizeof(int));
            cacheOut.write((char *) MRF_data.data(), sizeof(EnergyType) * MRF_data.size());
            cacheOut.close();
        }
        cout << "Done" << endl;
        //compute unary disparity
        unaryDisp.initialize(width, height, -1);
        for(auto i=0; i<width * height; ++i){
            EnergyType min_e = std::numeric_limits<EnergyType>::max();
            for(auto d=0; d<dispResolution; ++d){
                if(MRF_data[dispResolution * i + d] < min_e){
                    unaryDisp.setDepthAtInd(i, (double)d);
                    min_e = MRF_data[dispResolution * i + d];
                }
            }
        }
    }

    void SceStereo::runStereo() {
        char buffer[1024] = {};
        initMRF();

        sprintf(buffer, "%s/temp/unaryDisp%05d.jpg", file_io.getDirectory().c_str(), anchor);
        unaryDisp.saveImage(string(buffer), 255.0 / (dispResolution) * 4);

        //ProposalSegPlnMeanshift proposalSegPlnMeanshift(file_io, images, unaryDisp, 0, );
    }
}//namespace sce_stereo