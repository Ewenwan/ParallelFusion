//
// Created by yanhang on 3/8/16.
//
#include "optimization.h"
#include "../external/QPBO1.4/QPBO.h"

using namespace std;
using namespace cv;
using namespace ParallelFusion;

namespace simple_stereo {

    double ParallelOptimize::optimize(stereo_base::Depth &result, const int max_iter) const {
        typedef LabelSpace<int> Space;
        typedef ParallelFusionPipeline<Space> Pipeline;

        result.initialize(width, height, -1);
        //configure as sequential fusion
        ParallelFusionOption option;
        option.num_threads = 1;
        option.probProposalFromOther = 0;
        option.max_iteration = 1;
        option.fuseSize = model.nLabel;

        Pipeline::GeneratorSet generators((size_t)option.num_threads);
        Pipeline::SolverSet solvers((size_t)option.num_threads);
        vector<Space> initials((size_t)option.num_threads);

        const int kPix = model.width * model.height;

        for(auto i=0; i<option.num_threads; ++i){
            initials[i].init(kPix);
            for(auto j=0; j<kPix; ++j)
                initials[i].getLabelOfNode(j).push_back(0);
            generators[i] = shared_ptr<ProposalGenerator<Space> >(new SimpleStereoGenerator(model.width * model.height, model.nLabel, 0));
            solvers[i] = shared_ptr<FusionSolver<Space> >(new SimpleStereoSolver(model));
        }

        Pipeline parallelFusionPipeline(option);

        float t = cv::getTickCount();
        printf("Start runing parallel optimization\n");
        parallelFusionPipeline.runParallelFusion(initials, generators, solvers);
        t = ((float)getTickCount() - t) / (float)getTickFrequency();

        SolutionType<LabelSpace<int> > solution;
        parallelFusionPipeline.getBestLabeling(solution);
        printf("Done! Final energy: %.5f\n", solution.first);

        for(auto i=0; i<model.width * model.height; ++i){
            result.setDepthAtInd(i, solution.second(i,0));
        }
        return 0.0;
    }

    void SimpleStereoGenerator::getProposals(LabelSpace<int> &proposals,
                                            const LabelSpace<int> &current_solution, const int N) {
        proposals.init(nPix, vector<int>((size_t)N, 0));
        for(auto i=0; i<N; ++i){
            printf("label %d\n", i);
            printf("getProposals: proposals.NumNode:%d\n", proposals.getNumNode());
            for(auto j=0; j<nPix; ++j)
                proposals(j,i) = nextLabel;
            nextLabel++;
        }
    }

    void SimpleStereoSolver::initSolver(const LabelSpace<int> &initial) {
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
        const int kProposal = (int) proposals.getLabelOfNode(0).size();

        for (auto i = 0; i < kProposal; ++i) {
            printf("Fusing proposal %d\n", i);
            bool graphCut = true;
            int label = proposals(0, i);
            for (auto nid = 1; nid < proposals.getNumNode(); ++nid) {
                if (proposals(nid, i) != label) {
                    graphCut = false;
                    break;
                }
            }
            if (graphCut) {
                //run alpha-expansion
                cout << "Running alpha-expansion..." << endl << flush;
                mrf->alpha_expansion(label);
                cout << "done" << endl <<flush;
            } else {
                //run QPBO
                kolmogorov::qpbo::QPBO<int> qpbo(kPix, 4 * kPix);
                qpbo.AddNode(kPix);
                for (auto j = 0; j < kPix; ++j)
                    qpbo.AddUnaryTerm(j, model(j, mrf->getLabel(j)), model(j, proposals(j, i)));

                for (auto y = 0; y < model.height - 1; ++y) {
                    for (auto x = 0; x < model.width - 1; ++x) {
                        int e00, e01, e10, e11;
                        int pix1 = y * model.width + x, pix2 = y * model.width + x + 1, pix3 =
                                (y + 1) * model.width + x;
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

                for (auto pix = 0; pix < kPix; ++pix) {
                    if (qpbo.GetLabel(pix) >= 0)
                        mrf->setLabel(pix, proposals(pix, i));
                }
            }
        }

        solution.init(kPix, vector<int>(1, 0));
        for (auto i = 0; i < kPix; ++i) {
            solution.getLabelOfNode(i)[0] = mrf->getLabel(i);
        }
        return 0;
    }

    double SimpleStereoSolver::evaluateEnergy(const LabelSpace<int> &solution) const {
        return 0;
    }
}