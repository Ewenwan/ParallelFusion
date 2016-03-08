#ifndef PARALLEL_FUSION_PIPELINE_H__
#define PARALLEL_FUSION_PIPELINE_H__

#include <memory>
#include <vector>
#include <limits>
#include <iostream>
#include <map>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <atomic>

//#include "FusionThread.h"
//#include "cv_utils/cv_utils.h"
#include "FusionSolver.h"
#include "thread_guard.h"
#include "LabelSpace.h"
#include "ProposalGenerator.h"


namespace ParallelFusion {
    template<typename T>
    using SolutionType = std::pair<double, std::vector<T> >;

    //synchronized solution type
    template<typename T>
    class SynSolution {
    public:
        void set(const SolutionType<T> &l) {
            std::lock_guard<std::mutex> lock(mt);
            solution = l;
        }

        void get(SolutionType<T> &v) const {
            std::lock_guard<std::mutex> lock(mt);
            v = solution;
        }

        double getEnergy() const{
            std::lock_guard<std::mutex> lock(mt);
            return solution.first;
        }

        SynSolution &operator=(const SynSolution &) = delete;

        SynSolution(SynSolution &) = delete;

    private:
        SolutionType<T> solution;
        std::mutex mt;
    };

    struct ParallelFusionOption {
        ParallelFusionOption() : probProposalFromOther(0.2), convergeThreshold(0.01), max_iteration(10), num_threads(6), fuseSize(2), addMethod(APPEND) { }
        enum ProposalAddition{APPEND, UNION};
        double probProposalFromOther;
        double convergeThreshold;
        int max_iteration;
        int num_threads;
        int fuseSize;
        ProposalAddition addMethod;
    };

    template<typename T, class SOLVER>
    class ParallelFusionPipeline {
        ParallelFusionPipeline(const ParallelFusionOption &option_) : option(option_), terminate(false) { }

        typedef std::shared_ptr<ProposalGenerator<T> > GeneratorPtr;
        typedef std::vector<GeneratorPtr> GeneratorSet;

        //run parallel fusion. The logic in this routine represents the master thread
        //input: num_threads: number of threads to use
        //       max_iter: maximum iterations of fusion in each thread
        //       initials: initial solution for each thread. The size must be num_threads
        //return: final energy
        //each thread has its own generator. Note, the size of generators must be the
        //same with option.num_thread. If all threads use exactly the same generator, duplicate
        //it multiple times
        double runParallelFusion(const std::vector<std::vector<T> > &initials,
                                 const GeneratorSet& generator);

        void getLabeling(std::vector<T>& solution);

        //slave threads
        void workerThread(const int id, const std::vector<T> &initial,
                          const GeneratorPtr &generator);

        inline const ParallelFusionOption &getOption() const { return option; }

        inline ParallelFusionOption &getOption() { return option; }


        ParallelFusionPipeline(ParallelFusionPipeline &) = delete;

        ParallelFusionPipeline &operator=(const ParallelFusionPipeline &) = delete;

    private:
        //store current best solutions from each thread. The array can be access by multiples threads
        //each solution store a vector of labeling, and corresponding energy
        std::vector<SynSolution<T> > bestSolutions;

        //The following two parameters controls the parallel fusion.
        //each fusion
        ParallelFusionOption option;

        int kSelfThread;
        int kOtherThread;

        //if one thread throws exception, notify all threads
        std::atomic<bool> terminate;
    };


    ///////////////////////////////////////////////////////////////////////

    template<typename T, class SOLVER>
    double ParallelFusionPipeline<T, SOLVER>::runParallelFusion(const std::vector<std::vector<T> > &initials,
                                                                const GeneratorSet& generators){
        CHECK_EQ(option.num_threads, initials.size());
        CHECK_EQ(option.num_threads, generators.size());

        bestSolutions.resize((size_t)option.num_threads);

        kOtherThread = (int)(option.fuseSize * option.probProposalFromOther);
        kSelfThread = option.fuseSize - kOtherThread;

        CHECK_GT(kSelfThread, 1) << "Probability of drawing proposals from other thread is too high";

        //launch threads and acquire futures for exception handling
        std::vector<thread_guard> slaves;
        for(auto tid=0; tid<slaves.size(); ++tid){
            std::thread t(&ParallelFusionPipeline::workerThread, this, tid, initials[tid], generators[tid]);
            slaves[tid].bind(t);
        }
    }

    template<typename T, class SOLVER>
    void ParallelFusionPipeline<T, SOLVER>::getLabeling(std::vector<T> &solution) {

    }

    template<typename T, class SOLVER>
    void ParallelFusionPipeline<T, SOLVER>::workerThread(const int id,
                                                         const std::vector<T> &initial,
                                                         const GeneratorPtr& generator){
        try {
            SOLVER solver;

            std::default_random_engine seed;
            std::uniform_int_distribution<int> distribution(0, option.num_threads);

            bool converge = false;

            SolutionType<T> current_solution;
            current_solution.first = solver.evaluateEnergy(initial);
            current_solution.second = initial;
            bestSolutions[id].set(current_solution);
            double lastEnergy = current_solution.first;

            for (auto iter = 0; iter < option.max_iteration; ++iter) {
                if(terminate.load()){
                    printf("Thread %d quited\n", id);
                    return;
                }
                LabelSpace<T> proposals;
                //generate proposal by own generator
                for(auto pid=0; pid < kSelfThread; ++pid) {
                    std::vector<T> curproposal;
                    generator->getProposals(curproposal, current_solution.second);
                    if(option.addMethod == ParallelFusionOption::APPEND)
                        proposals.appendSolution(curproposal);
                    else
                        proposals.unionSolution(curproposal);
                }
                //Take best solutions from other threads
                for(auto pid=0; pid < kOtherThread; ++pid) {
                    int tid = distribution(seed);
                    SolutionType<T> s;
                    bestSolutions[tid].get(s);
                    if(option.addMethod == ParallelFusionOption::APPEND)
                        proposals.appendSolution(s.second);
                    else
                        proposals.unionSolution(s.second);
                }

                //solve
                SolutionType<T> curSolution;
                solver.solve(proposals, curSolution.second);
                curSolution.first = solver.evaluateEnergy(curSolution.second);
                bestSolutions[id].set(curSolution);
                //double diffE =  lastEnergy - curSolution.first;
            }
        }catch(const std::exception& e){
            terminate.store(true);
            printf("Thread %d throws and exception: %s\n", id, e.what());
            return;
        }
    }

}//namespace ParallelFusion
#endif
