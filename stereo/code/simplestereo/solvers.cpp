//
// Created by yanhang on 3/10/16.
//

#include "optimization.h"

using namespace std;
using namespace cv;
using namespace ParallelFusion;

namespace simple_stereo{
    SimpleStereoGenerator::SimpleStereoGenerator(const int nPix_, const int startid_, const int interval_, const int num_, const bool randomOrder_):
            nPix(nPix_), randomOrder(randomOrder_), nextLabel(0){
        labelTable.resize((size_t)num_);
        CHECK(!labelTable.empty());
        labelTable[0] = startid_;
        for(auto i=1; i<labelTable.size(); ++i)
            labelTable[i] = labelTable[i-1] + interval_;
        if(randomOrder)
            std::random_shuffle(labelTable.begin(), labelTable.end());
    }

    void SimpleStereoGenerator::getProposals(CompactLabelSpace &proposals,
                                             const CompactLabelSpace &current_solution, const int N) {
        for(auto i=0; i<N; ++i){
            proposals.getSingleLabel().push_back(labelTable[nextLabel]);
            nextLabel = (nextLabel + 1) % (int)labelTable.size();
        }
    }

    void SimpleStereoSolver::initSolver(const CompactLabelSpace &initial) {
        CHECK_EQ(initial.getNumNode(), model->width * model->height);
        EnergyFunction *energy_function = new EnergyFunction(new DataCost(const_cast<int*>(model->MRF_data)),
                                                             new SmoothnessCost(1, 4, model->weight_smooth,
                                                                                const_cast<int*>(model->hCue),
                                                                                const_cast<int*>(model->vCue)));
        mrf = new Expansion(model->width, model->height, model->nLabel, energy_function);
        mrf->initialize();
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

        solution = current_solution;
        for(auto i=0; i<kPix; ++i)
            mrf->setLabel(i, current_solution.second(i, 0));
        for (auto i = 0; i < singleLabel.size(); ++i) {
            //printf("Fusing proposal with graph cut %d\n", singleLabel[i] );
            mrf->alpha_expansion(singleLabel[i]);
            //cout << "done" << endl << flush;
        }
        for(auto i=0; i<kPix; ++i)
            solution.second(i,0) = mrf->getLabel(i);

        for (auto i = 0; i < kFullProposal; ++i) {
            //run QPBO
            //printf("Running QPBO...\n");
            fuseTwoSolution(solution.second, proposals,i,model);
            //printf("Done. Unlabeled:%.3f, label changed:%.3f\n", unlabeled / (float)kPix, changed / (float)kPix);
        }
        solution.second.getSingleLabel().clear();
        solution.first = evaluateEnergy(solution.second);
    }

    double SimpleStereoSolver::evaluateEnergy(const CompactLabelSpace &solution) const {
        CHECK_EQ(solution.getNumNode(), kPix);
        double e = 0;
        const int w = model->width;
        const int h = model->height;
        const double r = model->MRFRatio;
        for(auto i=0; i<kPix; ++i) {
            e += model->MRF_data[i * model->nLabel + solution(i, 0)] / model->MRFRatio;
        }
        for(auto x=0; x<w - 1; ++x) {
            for (auto y = 0; y < h - 1; ++y) {
                int sc = model->computeSmoothCost(y * w + x, solution(y * w + x, 0), solution(y * w + x + 1, 0), true) +
                         model->computeSmoothCost(y * w + x, solution(y * w + x, 0), solution((y + 1) * w + x, 0), true);
                e += (double) sc / r;
            }
        }
        return e;
    }


    double fuseTwoSolution(CompactLabelSpace& s1, const CompactLabelSpace& s2, const int pid, const MRFModel<int>* model){
//        CHECK_EQ(s1.getNumNode(), s2.getNumNode());
//        CHECK_GT(s1.getNumNode(), 0);
//        CHECK_LT(pid, s2.getLabelSpace()[0].size());

        const int kPix = s1.getNumNode();
        kolmogorov::qpbo::QPBO<int> qpbo(kPix, kPix*2);
        qpbo.AddNode(kPix);
        for(auto i=0; i<kPix; ++i)
            qpbo.AddUnaryTerm(i, model->operator()(i, s1(i,0)), model->operator()(i,s2(i, pid)));
        for(auto x=0; x<model->width-1; ++x){
            for(auto y=0; y<model->height-1; ++y){
                int e00, e01, e10, e11;
                int pix1 = y * model->width + x, pix2 = y * model->width + x + 1, pix3 =
                        (y + 1) * model->width + x;
                int l10, l11, l12;
                l10 = s2(pix1, pid);
                l11 = s2(pix2, pid);
                l12 = s2(pix3, pid);
                //x direction
                e00 = model->computeSmoothCost(pix1, s1(pix1, 0), s1(pix2, 0), true);
                e01 = model->computeSmoothCost(pix1, s1(pix1, 0), l11, true);
                e10 = model->computeSmoothCost(pix1, l10, s1(pix2, 0), true);
                e11 = model->computeSmoothCost(pix1, l10, l11, true);
                qpbo.AddPairwiseTerm(pix1, pix2, e00, e01, e10, e11);

                //y direction
                e00 = model->computeSmoothCost(pix1, s1(pix1, 0), s1(pix3, 0), false);
                e01 = model->computeSmoothCost(pix1, s1(pix1, 0), l12, false);
                e10 = model->computeSmoothCost(pix1, l10, s1(pix3, 0), false);
                e11 = model->computeSmoothCost(pix1, l10, l12, false);
                qpbo.AddPairwiseTerm(pix1, pix3, e00, e01, e10, e11);
            }
        }

        qpbo.MergeParallelEdges();
        qpbo.Solve();
        qpbo.ComputeWeakPersistencies();

        for(auto i=0; i<kPix; ++i){
            if(qpbo.GetLabel(i) == 1)
                s1(i,0) = s2(i,pid);
        }
    }

    void dumpOutData(const ParallelFusionPipeline<CompactLabelSpace>& pipeline, const string& prefix){
        char buffer[1024] = {};
        const GlobalTimeEnergyProfile& profile = pipeline.getGlobalProfile();
        sprintf(buffer, "%s_global.txt", prefix.c_str());
        ofstream globalOut(buffer);
        CHECK(globalOut.is_open());
        globalOut << "Global\nTime\tEnergy" << endl;
        for(const auto& ob: profile.getProfile())
            globalOut << ob.first << '\t' << ob.second << endl;
        globalOut.close();

        const vector<list<Observation> >& threadProfiles = pipeline.getAllThreadProfiles();
        for(auto tid=0; tid < threadProfiles.size(); ++tid){
            sprintf(buffer, "%s_thread%d.txt", prefix.c_str(), tid);
            ofstream threadOut(buffer);
            CHECK(threadOut.is_open());
            threadOut << "Thread " << tid << endl << "Time\tEnergy" << endl;
            for(const auto& ob: threadProfiles[tid])
                threadOut << ob.first << '\t' << ob.second << endl;
            threadOut.close();
        }
    }
}//namespace simple_stereo