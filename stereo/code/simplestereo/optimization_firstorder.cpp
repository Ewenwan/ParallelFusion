//
// Created by yanhang on 3/3/16.
//

#include "optimization.h"
#include "../external/MRF2.2/mrf.h"
#include "../external/MRF2.2/GCoptimization.h"

using namespace std;
using namespace cv;
using namespace Eigen;

using namespace stereo_base;
namespace simple_stereo{

    double FirstOrderOptimize::optimize(Depth &result, const int max_iter) const {
        EnergyFunction *energy_function = new EnergyFunction(new DataCost(const_cast<int *>(model.MRF_data.data())),
                                                             new SmoothnessCost(1, 4, model.weight_smooth,
                                                                                const_cast<int *>(model.hCue.data()),
                                                                                const_cast<int *>(model.vCue.data())));
        shared_ptr<MRF> mrf(new Expansion(width, height, nLabel, energy_function));
        mrf->initialize();

        //randomly initialize
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, nLabel - 1);
        for (auto i = 0; i < width * height; ++i)
            mrf->setLabel(i, distribution(generator));

        double initDataEnergy = (double) mrf->dataEnergy() / model.MRFRatio;
        double initSmoothEnergy = (double) mrf->smoothnessEnergy() / model.MRFRatio;
        float t;
        mrf->optimize(max_iter, t);
        double finalDataEnergy = (double) mrf->dataEnergy() / model.MRFRatio;
        double finalSmoothEnergy = (double) mrf->smoothnessEnergy() / model.MRFRatio;

        printf("Graph cut finished.\nInitial energy: (%.3f, %.3f, %.3f)\nFinal energy: (%.3f,%.3f,%.3f)\nTime usage: %.2fs\n",
               initDataEnergy, initSmoothEnergy, initDataEnergy + initSmoothEnergy,
               finalDataEnergy, finalSmoothEnergy, finalDataEnergy + finalSmoothEnergy, t);

        result.initialize(width, height, -1);
        for(auto i=0; i<width * height; ++i)
            result.setDepthAtInd(i, mrf->getLabel(i));
    }

}//namespace dynamic_stereo