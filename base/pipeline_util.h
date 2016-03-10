//
// Created by yanhang on 3/10/16.
//

#ifndef PARALLELFUSION_PIPELINE_UTIL_H
#define PARALLELFUSION_PIPELINE_UTIL_H

#include "FusionSolver.h"
#include "thread_guard.h"
#include "LabelSpace.h"
#include "ProposalGenerator.h"
#include <list>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace ParallelFusion{

    //type for energy observation: <time, energy>
    using Observation = std::pair<double, double>;

    //synchronized solution type
    template<class LABELSPACE>
    class SynSolution {
    public:
        SynSolution(): solution(SolutionType<LABELSPACE>(-1, LABELSPACE())){}
        void set(const SolutionType<LABELSPACE> &l) {
            std::lock_guard<std::mutex> lock(mt);
            solution = l;
        }

        void get(SolutionType<LABELSPACE> &v) const {
            std::lock_guard<std::mutex> lock(mt);
            v = solution;
        }

        double getEnergy() const{
            std::lock_guard<std::mutex> lock(mt);
            return solution.first;
        }

        //the following two method can only be called from single thread!!!
        inline SolutionType<LABELSPACE>& getSolution(){
            return solution;
        }
        inline const SolutionType<LABELSPACE>& getSolution() const{
            return solution;
        }

        SynSolution &operator=(const SynSolution &) = delete;

        SynSolution(SynSolution & rhs) = delete;

    private:
        SolutionType<LABELSPACE> solution;
        mutable std::mutex mt;
    };

    class GlobalTimeEnergyProfile{
    public:
        GlobalTimeEnergyProfile(const bool keep_minimum_ = true): keep_minimum(keep_minimum_){}
        inline void addObservation(const Observation& ob){
            add(ob);
        }

        inline void addObservation(const double time, const double energy){
            Observation ob(time, energy);
            addObservation(ob);
        }

        //the following method can only be called from single thread!
        inline const std::list<Observation>& getProfile() const{
            return profile;
        }
    private:
        void add(const Observation& ob){
            std::lock_guard<std::mutex> lock(mt);
            if(profile.empty())
                profile.push_back(ob);
            else
            if(ob.second <= profile.back().second)
                profile.push_back(ob);
        }
        mutable std::mutex mt;
        const bool keep_minimum;
        std::list<Observation> profile;
    };

    struct ParallelFusionOption {
        ParallelFusionOption() : convergeThreshold(0.01), max_iteration(10), num_threads(6), synchronize(false), selectionMethod(RANDOM){ }
        //Addition method: how add two proposals. APPEND: simply append; UNION: take union, remove duplicate labels
        double convergeThreshold;
        int max_iteration;
        int num_threads;
        bool synchronize;


        enum SolutionSelection{RANDOM, BEST};
        SolutionSelection selectionMethod;
    };

    struct ThreadOption {
        ThreadOption() : kTotal(1), kOtherThread(0), solution_exchange_interval(1), is_monitor(false) { }
        int kTotal;
        int kOtherThread;
        int solution_exchange_interval;
        bool is_monitor;
    };

}//namespace ParallelFusion
#endif //PARALLELFUSION_PIPELINE_UTIL_H
