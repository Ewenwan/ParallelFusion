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
        typedef CompactLabelSpace Space;
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

        SolutionType<Space > solution;
        parallelFusionPipeline.getBestLabeling(solution);
        printf("Done! Final energy: %.5f, running time: %.3f\n", solution.first, t);

        for(auto i=0; i<model.width * model.height; ++i){
            result.setDepthAtInd(i, solution.second(i,0));
        }
        return solution.first;
    }

    void SimpleStereoGenerator::getProposals(CompactLabelSpace &proposals,
                                            const CompactLabelSpace &current_solution, const int N) {
        for(auto i=0; i<N; ++i){
            printf("label %d\n", i);
            printf("getProposals: proposals.NumNode:%d\n", proposals.getNumNode());
            proposals.getSingleLabel().push_back(nextLabel);
            nextLabel++;
        }
    }

    void SimpleStereoSolver::initSolver(const CompactLabelSpace &initial) {
        CHECK_EQ(initial.getNumNode(), model.width * model.height);
        EnergyFunction *energy_function = new EnergyFunction(new DataCost(const_cast<int *>(model.MRF_data.data())),
                                                             new SmoothnessCost(1, 4, model.weight_smooth,
                                                                                const_cast<int *>(model.hCue.data()),
                                                                                const_cast<int *>(model.vCue.data())));
        mrf = shared_ptr<Expansion>(new Expansion(model.width, model.height, model.nLabel, energy_function));
        mrf->initialize();
        mrf->clearAnswer();
        for(auto i=0; i<kPix; ++i)
            mrf->setLabel(i, initial(i,0));
    }

    void SimpleStereoSolver::solve(const CompactLabelSpace &proposals,
                                     const SolutionType<CompactLabelSpace>& current_solution,
                                     SolutionType<CompactLabelSpace>& solution){
        CHECK(!proposals.empty());
        int kFullProposal;
        if(proposals.getLabelSpace().empty())
            kFullProposal = 0;
        else
            kFullProposal = (int) proposals.getLabelOfNode(0).size();

        const vector<int> &singleLabel = proposals.getSingleLabel();
        for (auto i = 0; i < singleLabel.size(); ++i) {
            printf("Fusing proposal with graph cut%d\n", i);
            mrf->alpha_expansion(singleLabel[i]);
            cout << "done" << endl << flush;
        }
        for (auto i = 0; i < kFullProposal; ++i) {
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

        for (auto i = 0; i < kPix; ++i) {
            solution.first = (double)mrf->totalEnergy() / model.MRFRatio;
            solution.second.getLabelOfNode(i)[0] = mrf->getLabel(i);
        }
    }
}