//
// Created by yanhang on 3/3/16.
//

#ifndef SIMPLESTEREO_OPTIMIZATION_H
#define SIMPLESTEREO_OPTIMIZATION_H

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <ctime>

#include "../stereo_base/depth.h"
#include "../stereo_base/file_io.h"

#include "../external/MRF2.2/GCoptimization.h"
#include "../external/QPBO1.4/QPBO.h"
#include "../../../base/LabelSpace.h"
#include "../../../base/FusionSolver.h"
#include "../../../base/ParallelFusionPipeline.h"
#include "../../../base/ProposalGenerator.h"

namespace simple_stereo {
    template<typename T>
    struct MRFModel{
        MRFModel(): MRF_data(NULL), hCue(NULL), vCue(NULL), width(0), height(0), MRFRatio(100.0){}
        ~MRFModel(){
            clear();
        }

        int width;
        int height;
        int nLabel;
        T* MRF_data;
        T* hCue;
        T* vCue;
        T weight_smooth;
        double MRFRatio;

        inline void clear(){
            delete MRF_data;
            delete hCue;
            delete vCue;
            MRF_data = NULL;
            hCue = NULL;
            vCue = NULL;
            width = 0;
            height = 0;
            nLabel = 0;
        }

        inline T operator()(int pixId, int l) const{
//            CHECK_LT(pixId, width * height);
//            CHECK_LT(l, nLabel);
            return MRF_data[pixId * nLabel + l];
        }
        void init(const int w, const int h, const int n, const double wei){
            MRF_data = new T[w * h * n];
            hCue = new T[w * h];
            vCue = new T[w * h];
            width = w;
            height = h;
            nLabel = n;
            weight_smooth = (T)(wei * MRFRatio);
            MRFRatio = 100.0;
        }

        inline T computeSmoothCost(const int pix, const int l1, const int l2, bool direction) const{
            double cue = direction ? hCue[pix] : vCue[pix];
            return (T)((double)weight_smooth * (std::min(4, std::abs(l1-l2))) * cue);
        }
    };

    class CompactLabelSpace: public ParallelFusion::LabelSpace<int>{
    public:
        inline size_t getNumSingleLabel() const{
            return singleLabel.size();
        }

        std::vector<int>& getSingleLabel(){
            return singleLabel;
        }

        const std::vector<int>& getSingleLabel() const{
            return singleLabel;
        }

        const void appendSpace(const CompactLabelSpace& rhs){
            if(num_nodes_ == 0){
                label_space_.resize((size_t)rhs.getNumNode());
                num_nodes_ = (int)label_space_.size();
            }
            CHECK_EQ(rhs.getNumNode(), getNumNode());
            for(auto i=0; i<label_space_.size(); ++i){
                for(auto j=0; j<rhs.getLabelOfNode(i).size(); ++j)
                    label_space_[i].push_back(rhs(i,j));
            }
            for(auto i=0; i<rhs.getSingleLabel().size(); ++i)
                singleLabel.push_back(rhs.getSingleLabel()[i]);
        }

        inline virtual bool empty() const{
            return singleLabel.empty() && label_space_.empty();
        }

        inline virtual void clear(){
            ParallelFusion::LabelSpace<int>::clear();
            singleLabel.clear();
        }
    private:
        std::vector<int> singleLabel;
    };

    class StereoOptimizer{
    public:
        StereoOptimizer(const stereo_base::FileIO& file_io_, const MRFModel<int>* model_): file_io(file_io_), model(model_),
                                                                                      width(model_->width), height(model_->height), nLabel(model_->nLabel){}
        virtual double optimize(stereo_base::Depth& result, const int max_iter) const = 0;
        double evaluateEnergy(const std::vector<int>& labeling) const{
            return 0;
        }
        double evaluateEnergy(const stereo_base::Depth& result) const{
            return 0;
        }

    protected:
        const stereo_base::FileIO& file_io;
        const MRFModel<int>* model;
        const int width;
        const int height;
        const int nLabel;
    };

    class FirstOrderOptimize: public StereoOptimizer{
    public:
        FirstOrderOptimize(const stereo_base::FileIO &file_io_, const MRFModel<int>* model_): StereoOptimizer(file_io_, model_){}
        virtual double optimize(stereo_base::Depth &result, const int max_iter) const;
    };

    class ParallelOptimize: public StereoOptimizer{
    public:
        ParallelOptimize(const stereo_base::FileIO &file_io_, const MRFModel<int>* model_, const int num_threads_):
                StereoOptimizer(file_io_, model_), num_threads(num_threads_){}
        virtual double optimize(stereo_base::Depth &result, const int max_iter) const;
    private:
        const int num_threads;
    };

    class HierarchyOptimize: public StereoOptimizer {
    public:
        HierarchyOptimize(const stereo_base::FileIO &file_io_, const MRFModel<int> *model_, const int num_threads_):
                StereoOptimizer(file_io_, model_), num_threads(num_threads_){}
        virtual double optimize(stereo_base::Depth& result, const int max_iter) const;
    private:
        const int num_threads;
    };

    class DummyGenerator: public ParallelFusion::ProposalGenerator<CompactLabelSpace>{
    public:
        virtual void getProposals(CompactLabelSpace& proposals, const CompactLabelSpace& current_solution, const int N){}
    };

    class SimpleStereoSolver : public ParallelFusion::FusionSolver<CompactLabelSpace> {
    public:
        SimpleStereoSolver(const MRFModel<int>* model_): model(model_), kPix(model->width * model->height){}
        ~SimpleStereoSolver(){
            delete mrf;
//            delete qpbo;
        }
        virtual void initSolver(const CompactLabelSpace& initial);
        virtual void solve(const CompactLabelSpace &proposals, const ParallelFusion::SolutionType<CompactLabelSpace>& current_solution,
                           ParallelFusion::SolutionType<CompactLabelSpace>& solution);
        virtual double evaluateEnergy(const CompactLabelSpace& solution) const;

    protected:
        const MRFModel<int>* model;
        const int kPix;
        Expansion* mrf;
        //kolmogorov::qpbo::QPBO<int>* qpbo;
    };

    class HierarchyStereoSolver: public SimpleStereoSolver{
    public:
        HierarchyStereoSolver(const MRFModel<int>* model_): SimpleStereoSolver(model_){}
        virtual void solve(const CompactLabelSpace &proposals, const ParallelFusion::SolutionType<CompactLabelSpace>& current_solution,
                           ParallelFusion::SolutionType<CompactLabelSpace>& solution);
    };

    class SimpleStereoMonitor: public ParallelFusion::FusionSolver<CompactLabelSpace>{
    public:
        typedef std::pair<double, double> Observation;

        SimpleStereoMonitor(const MRFModel<int>* model_): model(model_), kPix(model->width * model->height){}

        virtual void initSolver(const CompactLabelSpace & initial){
            observations.clear();
            std::time(&start_time);
            t = cv::getTickCount();
        }
        virtual double evaluateEnergy(const CompactLabelSpace & solution) const;

        virtual void solve(const CompactLabelSpace &proposals, const ParallelFusion::SolutionType<CompactLabelSpace>& current_solution,
                           ParallelFusion::SolutionType<CompactLabelSpace>& solution);

        void writePlot(const std::string& path) const;
    protected:
        const MRFModel<int>* model;
        const int kPix;
        std::time_t start_time;
        float t;
        std::vector<Observation> observations;
    };

    class SimpleStereoMonitorFusion: public SimpleStereoMonitor{
    public:
        SimpleStereoMonitorFusion(const MRFModel<int>* model_): SimpleStereoMonitor(model_){}
        virtual void solve(const CompactLabelSpace &proposals, const ParallelFusion::SolutionType<CompactLabelSpace>& current_solution,
                           ParallelFusion::SolutionType<CompactLabelSpace>& solution);
    };

    class SimpleStereoGenerator: public ParallelFusion::ProposalGenerator<CompactLabelSpace>{
    public:
        SimpleStereoGenerator(const int nPix_, const int startid_, const int interval_, const int num_, const bool randomOrder_ = true);
        virtual void getProposals(CompactLabelSpace& proposals, const CompactLabelSpace& current_solution, const int N);
    private:
        const int nPix;
        const bool randomOrder;
        std::vector<int> labelTable;
        int nextLabel;
    };


    double fuseTwoSolution(CompactLabelSpace& s1, const CompactLabelSpace& s2, const int pid, const MRFModel<int>* model);
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_OPTIMIZATION_H
