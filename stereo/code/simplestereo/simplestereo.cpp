//
// Created by yanhang on 3/7/16.
//

#include "../external/MRF2.2/mrf.h"
#include "../external/QPBO1.4/QPBO.h"
#include "simplestereo.h"
#include "../stereo_base/local_matcher.h"

using namespace std;
using namespace stereo_base;
using namespace cv;
using namespace Eigen;
using namespace ParallelFusion;

namespace simple_stereo {
    void SimpleStereoGenerator::getProposal(LabelSpace<int>& proposals,
                                            const LabelSpace<int> &current_solution, const int N) {
//        proposal.resize(current_solution.size());
//        for(auto& v: proposal)
//            v = nextLabel;
//        nextLabel = (nextLabel + 1) % nLabel;
    }

    void SimpleStereoSolver::initSolver(const LabelSpace<int>& initial) {
        CHECK_EQ(initial.getNumNode(), model.width * model.height);
        EnergyFunction *energy_function = new EnergyFunction(new DataCost(const_cast<int *>(model.MRF_data.data())),
                                                             new SmoothnessCost(1, 4, model.weight_smooth,
                                                                                const_cast<int *>(model.hCue.data()),
                                                                                const_cast<int *>(model.vCue.data())));
        mrf = shared_ptr<Expansion>(new Expansion(model.width, model.height, energy_function));
    }

    double SimpleStereoSolver::solve(const LabelSpace<int> &proposals,
                                     LabelSpace<int> &solution) const {
        CHECK(!proposals.empty());
        CHECK_EQ(proposals.getNumNode(), kPix);
        const int kProposal = (int)proposals.getLabelOfNode(0).size();

        for(auto i=0; i<kProposal; ++i){
            bool graphCut = true;
            int label = proposals(0, i);
            for(auto nid=1; nid < proposals.getNumNode(); ++nid){
                if(proposals(nid, i) != label){
                    graphCut = false;
                    break;
                }
            }
            if(graphCut){
                //run alpha-expansion
                int e = mrf->alpha_expansion(label);
            }else{
                //run QPBO
                kolmogorov::qpbo::QPBO qpbo<int>(kPix, 4*kPix);
                qpbo.AddNode(kPix);
                for(auto j=0; j<kPix; ++j)
                    qpbo.AddUnaryTerm(j, model(j, mrf->getLabel(j)), model(j, proposals(j, i)));

                for(auto y=0; y<model.height-1; ++y){
                    for(auto x=0; x<model.width-1; ++x){
                        int e00, e01, e10, e11;
                        int pix1 = y*model.width+x, pix2 = y*model.width+x+1, pix3 = (y+1)*model.width+x;
                        //x direction
                        e00 = smoothnessCost(pix1, mrf->getLabel(pix1), mrf->getLabel(pix2), true);
                        e01 = smoothnessCost(pix1, mrf->getLabel(pix1), proposals(pix2, i), true);
                        e10 = smoothnessCost(pix1, proposals(pix1, i), mrf->getLabel(pix2), true);
                        e11 = smoothnessCost(pix1, proposals(pix1, i), proposals(pix2, i), true);
                        qpbo.AddPairwiseTerm(pix1, pix2, e00, e01, e10, e11);

                        //y direction
                        e00 = smoothnessCost(pix1, mrf->getLabel(pix1), mrf->getLabel(pix3), false);
                        e01 = smoothnessCost(pix1, mrf->getLabel(pix1), proposals(pix3, i), false);
                        e10 = smoothnessCost(pix1, proposals(pix1, i), mrf->getLabel(pix3), false);
                        e11 = smoothnessCost(pix1, proposals(pix1, i), proposals(pix3, i), false);
                        qpbo.AddPairwiseTerm(pix1, pix3, e00, e01, e10, e11);
                    }
                }

                qpbo.Solve();
                qpbo.ComputeWeakPersistencies();

                for(auto pix=0; pix<kPix; ++pix){
                    if(qpbo.GetLabel(pix) >= 0)
                        mrf->setLabel(pix, proposals(pix, i));
                }
            }
        }

        return 0;
    }

    double SimpleStereoSolver::evaluateEnergy(const std::vector<int> &solution) const {
        return 0;
    }

    SimpleStereo::SimpleStereo(const FileIO &file_io_, const int anchor_, const int dispResolution_, const double weight_smooth_):
            file_io(file_io_), anchor(anchor_){
        CHECK_GE(file_io.getTotalNum(), 2) << "Too few images at " << file_io.getDirectory();
        images.resize((size_t)file_io.getTotalNum());
        for(auto i=0; i<images.size(); ++i)
            images[i] = imread(file_io.getImage(i));
        width = images[0].cols;
        height = images[0].rows;

        model.init(width, height, dispResolution_, weight_smooth_);
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
        ifstream fin(buffer, ios::binary);
        bool recompute = true;
        if (fin.is_open()) {
            int frame, resolution, type;
            fin.read((char *) &frame, sizeof(int));
            fin.read((char *) &resolution, sizeof(int));
            fin.read((char *) &type, sizeof(int));
            printf("Cached data: anchor:%d, resolution:%d, Energytype:%d\n",
                   frame, resolution, type);
            if (frame == anchor && resolution == model.nLabel  &&
                type == sizeof(int)) {
                printf("Reading unary term from cache...\n");
                fin.read((char *) model.MRF_data.data(), model.MRF_data.size() * sizeof(int));
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
                    for (int d = 0; d < model.nLabel; ++d) {
                        //project onto other views and compute matching cost
                        vector<vector<double>> patches(images.size());
                        for (auto v = 0; v < images.size(); ++v) {
                            double distance = (double) (v - anchor) * (double)d;
                            Vector2d imgpt(x - distance, y);
                            local_matcher::samplePatch(images[v], imgpt, 3, patches[v]);
                        }
                        double mCost = local_matcher::sumMatchingCost(patches, anchor);
                        model.MRF_data[model.nLabel * (y * width + x) + d] = (int) ((mCost + 1) * model.MRFRatio);
                    }
                }
            }

            const int energyTypeSize = sizeof(EnergyType);
            ofstream cacheOut(buffer, ios::binary);
            cacheOut.write((char *) &anchor, sizeof(int));
            cacheOut.write((char *) &model.nLabel, sizeof(int));
            cacheOut.write((char *) &energyTypeSize, sizeof(int));
            cacheOut.write((char *) model.MRF_data.data(), sizeof(EnergyType) * model.MRF_data.size());
            cacheOut.close();
        }
        cout << "Done" << endl;
        //compute unary disparity
        unaryDisp.initialize(width, height, -1);
        for(auto i=0; i<width * height; ++i){
            EnergyType min_e = std::numeric_limits<EnergyType>::max();
            for(auto d=0; d<model.nLabel; ++d){
                if(model.MRF_data[model.nLabel * i + d] < min_e){
                    unaryDisp.setDepthAtInd(i, (double)d);
                    min_e = model.MRF_data[model.nLabel * i + d];
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
                        model.vCue[y * width + x] = 0;
                    else
                        model.vCue[y * width + x] = (EnergyType) ((diff - t) * (diff - t) * ratio);
                }
                if (x < width - 1) {
                    Vec3b pix2 = img.at<Vec3b>(y, x + 1);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]) / 255.0;
                    double diff = (dpix1 - dpix2).norm();
                    if (diff > t)
                        model.hCue[y * width + x] = 0;
                    else
                        model.hCue[y * width + x] = (EnergyType) ((diff - t) * (diff - t) * ratio);
                }
            }
        }
    }


    void SimpleStereo::runStereo() {
        char buffer[1024] = {};
        initMRF();

        sprintf(buffer, "%s/temp/unaryDisp.jpg", file_io.getDirectory().c_str());
        unaryDisp.saveImage(buffer, 256.0 / (double)model.nLabel);
    }
}
