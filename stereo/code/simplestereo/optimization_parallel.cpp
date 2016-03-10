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
        ParallelFusionOption pipelineOption;
        pipelineOption.num_threads = 2;
        pipelineOption.max_iteration = 512;
        const int kLabelPerThread = model->nLabel / pipelineOption.num_threads;

        Pipeline::GeneratorSet generators((size_t)pipelineOption.num_threads);
        Pipeline::SolverSet solvers((size_t)pipelineOption.num_threads);
        vector<Space> initials((size_t)pipelineOption.num_threads);
        vector<ThreadOption> threadOptions((size_t)pipelineOption.num_threads);

        const int kPix = model->width * model->height;

        //slave threads
        for(auto i=0; i<pipelineOption.num_threads - 1; ++i){
            const int startid = i;
            const int interval = pipelineOption.num_threads;
            initials[i].init(kPix, vector<int>(1, startid));
            threadOptions[i].kOtherThread = 0;
            threadOptions[i].kSelfThread = 1;
            printf("Thread %d, start: %d, interval:%d, num:%d\n", i, startid, pipelineOption.num_threads, kLabelPerThread);
            generators[i] = shared_ptr<ProposalGenerator<Space> >(new SimpleStereoGenerator(model->width * model->height, startid, interval, kLabelPerThread));
            solvers[i] = shared_ptr<FusionSolver<Space> >(new SimpleStereoSolver(model));
            printf("Initial energy on thread %d: %.5f\n", i, solvers[i]->evaluateEnergy(initials[i]));
        }

        //monitor thread
        initials.back().init(0);
        threadOptions.back().is_monitor = true;
        threadOptions.back().kSelfThread = 0;
        generators.back() = shared_ptr<ProposalGenerator<Space> >(new DummyGenerator());
        solvers.back() = shared_ptr<FusionSolver<Space> >(new SimpleStereoMonitor(model));

        Pipeline parallelFusionPipeline(pipelineOption);

        time_t start_t, end_t;
        time(&start_t);
        float t = (float)getTickCount();
        printf("Start runing parallel optimization\n");
        parallelFusionPipeline.runParallelFusion(initials, generators, solvers, threadOptions);
        time(&end_t);
        t = ((float)getTickCount() - t) / (float)getTickFrequency();

        SolutionType<Space > solution;
        parallelFusionPipeline.getBestLabeling(solution);
        printf("Done! Final energy: %.5f, running time: %.3fs\n", solution.first, t);

        char buffer[1024] = {};
        sprintf(buffer, "%s/temp/plot.txt", file_io.getDirectory().c_str());
        const SimpleStereoMonitor* monitorPtr = dynamic_cast<SimpleStereoMonitor*>(solvers.back().get());
        monitorPtr->writePlot(string(buffer));

        for(auto i=0; i<model->width * model->height; ++i){
            result.setDepthAtInd(i, solution.second(i,0));
        }
        return solution.first;
    }


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

//        qpbo = new kolmogorov::qpbo::QPBO<int>(kPix, 2 * kPix);
//        qpbo->AddNode(kPix);
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
            kolmogorov::qpbo::QPBO<int> qpbo(kPix, kPix*2);
            qpbo.AddNode(kPix);
            for (auto j = 0; j < kPix; ++j) {
                qpbo.AddUnaryTerm(j, model->operator()(j, solution.second(j, 0)),
                                  model->operator()(j, proposals(j, i)));
            }

            for (auto y = 0; y < model->height - 1; ++y) {
                for (auto x = 0; x < model->width - 1; ++x) {
                    int e00, e01, e10, e11;
                    int pix1 = y * model->width + x, pix2 = y * model->width + x + 1, pix3 =
                            (y + 1) * model->width + x;
                    int l10, l11, l12;
                    l10 = proposals(pix1, i);
                    l11 = proposals(pix2, i);
                    l12 = proposals(pix3, i);
                    //x direction
                    e00 = smoothnessCost(pix1, solution.second(pix1, 0), solution.second(pix2, 0), true);
                    e01 = smoothnessCost(pix1, solution.second(pix1, 0), l11, true);
                    e10 = smoothnessCost(pix1, l10, solution.second(pix2, 0), true);
                    e11 = smoothnessCost(pix1, l10, l11, true);
                    qpbo.AddPairwiseTerm(pix1, pix2, e00, e01, e10, e11);

                    //y direction
                    e00 = smoothnessCost(pix1, solution.second(pix1, 0), solution.second(pix3, 0), false);
                    e01 = smoothnessCost(pix1, solution.second(pix1, 0), l12, false);
                    e10 = smoothnessCost(pix1, l10, solution.second(pix3, 0), false);
                    e11 = smoothnessCost(pix1, l10, l12, false);
                    qpbo.AddPairwiseTerm(pix1, pix3, e00, e01, e10, e11);
                }
            }
            qpbo.MergeParallelEdges();
            qpbo.Solve();
            qpbo.ComputeWeakPersistencies();
//          float unlabeled = 0;
//            float changed = 0;
            for (auto pix = 0; pix < kPix; ++pix) {
                if (qpbo.GetLabel(pix) == 1) {
                    solution.second(pix, 0) = proposals(pix, i);
//                    changed += 1.0;
                }
//                if(qpbo.GetLabel(pix) == -1)
//                    unlabeled += 1.0;
            }
            //printf("Done. Unlabeled:%.3f, label changed:%.3f\n", unlabeled / (float)kPix, changed / (float)kPix);
        }

        //solution.first = evaluateEnergy(solution.second);
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
                int sc = smoothnessCost(y * w + x, solution(y * w + x, 0), solution(y * w + x + 1, 0), true) +
                         smoothnessCost(y * w + x, solution(y * w + x, 0), solution((y + 1) * w + x, 0), true);
                e += (double) sc / r;
            }
        }
        return e;
    }


    void SimpleStereoMonitor::solve(const CompactLabelSpace &proposals, const ParallelFusion::SolutionType<CompactLabelSpace>& current_solution,
                       ParallelFusion::SolutionType<CompactLabelSpace>& solution){
        CHECK(!proposals.getLabelSpace().empty());
        time_t current_t;
        std::time(&current_t);
        float difft = ((float)getTickCount() - t) / (float)getTickFrequency();
        const size_t nSolution = proposals.getLabelSpace()[0].size();
        double min_energy = numeric_limits<double>::max();
        for(auto i=0; i<nSolution; ++i){
            CompactLabelSpace curSolution;
            curSolution.init(kPix, vector<int>(1,0));
            for(auto j=0; j<kPix; ++j)
                curSolution(j,0) = proposals(j,i);
            double curE = evaluateEnergy(curSolution);
            if(curE < min_energy)
                min_energy = curE;
        }

        //observations.push_back(Observation(std::difftime(current_t, start_time), min_energy));
        observations.push_back(Observation(difft, min_energy));
    }

    double SimpleStereoMonitor::evaluateEnergy(const CompactLabelSpace &solution) const {
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
                int sc = smoothnessCost(y * w + x, solution(y * w + x, 0), solution(y * w + x + 1, 0), true) +
                         smoothnessCost(y * w + x, solution(y * w + x, 0), solution((y + 1) * w + x, 0), true);
                e += (double) sc / r;
            }
        }
        return e;
    }
    void SimpleStereoMonitor::writePlot(const std::string &path) const {
        ofstream fout(path.c_str());
        CHECK(fout.is_open());
        fout << "Time\tEnergy" << endl;
        char buffer[1024] = {};
        for(const auto& ob: observations){
            sprintf(buffer, "%.3f\t%.5f\n", ob.first, ob.second);
            fout << buffer;
        }

        fout.close();
    }
}