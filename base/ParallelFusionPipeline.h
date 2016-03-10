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
#include <list>
#include <mutex>
#include <opencv2/opencv.hpp>



namespace ParallelFusion {

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

//    template<typename T>
//    class SynVector{
//    public:
//        SynVector(const size_t size):data.resize(size){}
//        const T get(int id) const{
//            CHECK_LT(id, data.size());
//            std::lock_guard<std::mutex> lock(mt);
//            return data[id];
//        }
//        void set(int id, const T& v){
//            CHECK_LT(id, data.size());
//            std::lock_guard<std::mutex> lock(mt);
//            data[id] = v;
//        }
//    private:
//        std::vector<T> data;
//        mutable std::mutex mt;
//    };

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

    template<class LABELSPACE>
    class ParallelFusionPipeline {
    public:
        //solver should be read only
        ParallelFusionPipeline(const ParallelFusionOption &option_) : option(option_), bestSolutions((size_t)option_.num_threads),
                                                                      terminate(false), write_flag((size_t)option_.num_threads), threadProfile((size_t)option_.num_threads){ }

        typedef std::shared_ptr<ProposalGenerator<LABELSPACE> > GeneratorPtr;
        typedef std::vector<GeneratorPtr> GeneratorSet;
        typedef std::shared_ptr<FusionSolver<LABELSPACE> > SolverPtr;
        typedef std::vector<SolverPtr> SolverSet;

        //run parallel fusion. The logic in this routine represents the master thread
        //input: num_threads: number of threads to use
        //       max_iter: maximum iterations of fusion in each thread
        //       initials: initial solution for each thread. The size must be num_threads
        //       gnerators: shared pointers of ProposalGenerator, one for each thread
        //       solvers: shared pointers of FusionSolver, one for each thread
        //return: final energy
        double runParallelFusion(const std::vector<LABELSPACE> &initials,
                                 const GeneratorSet& generators,
                                 const SolverSet& solvers,
                                 const std::vector<ThreadOption> &thread_options);

        double getBestLabeling(SolutionType<LABELSPACE>& solution) const;
        void getAllResult(LABELSPACE& solutions) const;

        //slave threads
        void workerThread(const int id,
                          const LABELSPACE& initial,
                          GeneratorPtr generator,
                          SolverPtr solver,
                          const ThreadOption &thread_option);
        void monitorThread(const int id, SolverPtr solver);

        inline const ParallelFusionOption &getOption() const { return option; }

        inline ParallelFusionOption &getOption() { return option; }

        ParallelFusionPipeline(ParallelFusionPipeline &) = delete;

        ParallelFusionPipeline &operator=(const ParallelFusionPipeline &) = delete;

    private:
        //store current best solutions from each thread. The array can be access by multiples threads
        //each solution store a vector of labeling, and corresponding energy
        std::vector<SynSolution<LABELSPACE> > bestSolutions;

        ParallelFusionOption option;
        //if one thread throws exception, notify all threads
        std::atomic<bool> terminate;
        //In the presence of monitor thread, all threads might need to be synchronized.
        std::vector<std::atomic<bool> > write_flag;

        std::vector<int> slaveThreadIds;
        std::vector<int> monitorThreadIds;

        //keep tracking of <time, energy> of each thread
        std::vector<std::list<Observation> > threadProfile;
    };


    ///////////////////////////////////////////////////////////////////////

    template<class LABELSPACE>
    double ParallelFusionPipeline<LABELSPACE>::runParallelFusion(const std::vector<LABELSPACE> &initials,
                                                                 const GeneratorSet& generators,
                                                                 const SolverSet& solvers,
                                                                 const std::vector<ThreadOption> &thread_options){

        CHECK_EQ(option.num_threads, initials.size());
        CHECK_EQ(option.num_threads, generators.size());
        CHECK_EQ(option.num_threads, solvers.size());
        CHECK_EQ(option.num_threads, thread_options.size());

        bool monitor_exists = false;

        //sanity checks
        for(auto i=0; i<option.num_threads; ++i){
            if(thread_options[i].is_monitor){
                monitor_exists = true;
                CHECK_EQ(thread_options[i].kTotal, 0) << "Monitor thread can not generate proposal";
                monitorThreadIds.push_back(i);
            }else{
                slaveThreadIds.push_back(i);
            }
            CHECK_GE(thread_options[i].kTotal, 0) << "Negative number of proposals from self.";
            CHECK_GE(thread_options[i].kOtherThread, 0) << "Negative number of proposals from others.";
            CHECK_GE(thread_options[i].solution_exchange_interval, 1) << "Solution exchange interval should be a positive number.";
        }
        //if synchronization is needed, there must be an monitor thread
        if(option.synchronize)
            CHECK(monitor_exists);

        //initialize arrays. Slave threads are store in 0..slaveThreadIds.size() elements
        for(auto i=0; i<slaveThreadIds.size(); ++i){
            const int& idx = slaveThreadIds[i];
            bestSolutions[i].set(SolutionType<LABELSPACE>(-1, initials[idx]));
            write_flag[i].store(true);
        }

        //launch slave threads. Join method is called from the destructor of thread_guard
        std::vector<thread_guard> slaves(slaveThreadIds.size());
        for(auto tid=0; tid<slaves.size(); ++tid){
            printf("Lauching slave threads %d...\n", tid);
            const int& idx = slaveThreadIds[tid];
            std::thread t(&ParallelFusionPipeline::workerThread, this, tid, std::ref(initials[idx]), generators[idx],
                          solvers[idx], thread_options[idx]);
            slaves[tid].bind(t);
        }

        //launch monitor threads
        std::vector<thread_guard> monitors(monitorThreadIds.size());
        for(auto tid=0; tid<monitorThreadIds.size(); ++tid){
            printf("Lauching monitor threads %d...\n", tid);
            const int& idx = monitorThreadIds[tid];
            std::thread t(&ParallelFusionPipeline::monitorThread, this, tid, solvers[idx]);
            monitors[tid].bind(t);
        }

        //wait for all slave threads to finish job
        for(auto i=0; i<slaves.size(); ++i)
            slaves[i].join();

        //quit monitor thread
        terminate.store(true);
    }

    template<class LABELSPACE>
    double ParallelFusionPipeline<LABELSPACE>::getBestLabeling(SolutionType<LABELSPACE> &solution) const {
        double minE = std::numeric_limits<double>::max();
        for(auto i=0; i<slaveThreadIds.size(); ++i){
            if(bestSolutions[i].getSolution().first < minE){
                solution.second.clear();
                solution.second.appendSpace(bestSolutions[i].getSolution().second);
                solution.first = bestSolutions[i].getSolution().first;
                minE = solution.first;
            }
        }
        return minE;
    }

    template<class LABELSPACE>
    void ParallelFusionPipeline<LABELSPACE>::workerThread(const int id,
                                                          const LABELSPACE& initial,
                                                          GeneratorPtr generator,
                                                          SolverPtr solver,
                                                          const ThreadOption &thread_option){
        try {
            printf("Thread %d lauched\n", id);
            std::default_random_engine seed;
            std::uniform_int_distribution<int> distribution(1, (int)slaveThreadIds.size() - 1);
            bool converge = false;

            solver->initSolver(initial);

            SolutionType<LABELSPACE> current_solution;
            current_solution.first = solver->evaluateEnergy(initial);
            current_solution.second = initial;
            bestSolutions[id].set(current_solution);

            double lastEnergy = current_solution.first;

            for(int iter=0; iter < option.max_iteration; ++iter) {
                if(terminate.load()){
                    printf("Thread %d quited\n", id);
                    return;
                }
                LABELSPACE proposals;
                //generate proposal by own generator
                LABELSPACE proposals_self;
                //printf("Generating proposals...\n");
                const int num_proposals_from_others = (iter + 1) % thread_option.solution_exchange_interval == 0 ? thread_option.kOtherThread : 0;
                printf("In iteration %d, thread %d generates %d proposals and grab %d solutions (from best solutions = %d)\n", iter, id, thread_option.kTotal - num_proposals_from_others, num_proposals_from_others, option.selectionMethod == ParallelFusionOption::BEST);
                generator->getProposals(proposals_self, current_solution.second, thread_option.kTotal - num_proposals_from_others);
                proposals.appendSpace(proposals_self);

                if (option.selectionMethod == ParallelFusionOption::RANDOM) {
                    for(auto pid=0; pid < num_proposals_from_others; ++pid) {
                        int idshift = distribution(seed);
                        int tid = (id + idshift) % (int)slaveThreadIds.size();
                        SolutionType<LABELSPACE> s;
                        bestSolutions[tid].get(s);
                        proposals.appendSpace(s.second);
                    }
                } else if (option.selectionMethod == ParallelFusionOption::BEST) { //
                    std::vector<std::pair<double, int> > solution_energy_index_pairs(slaveThreadIds.size());
                    for(auto tid=0; tid < slaveThreadIds.size(); ++tid)
                        solution_energy_index_pairs[tid] = std::make_pair(bestSolutions[tid].getEnergy(), tid);
                    sort(solution_energy_index_pairs.begin(), solution_energy_index_pairs.end());
                    for(auto pid=0; pid < num_proposals_from_others; ++pid) {
                        int tid = solution_energy_index_pairs[pid].second;
                        SolutionType<LABELSPACE> s;
                        bestSolutions[tid].get(s);
                        proposals.appendSpace(s.second);
                    }
                }

                //printf("Solving...\n");
                SolutionType<LABELSPACE> curSolution;
                solver->solve(proposals, current_solution, curSolution);
                //printf("Done. Energy: %.5f\n", curSolution.first);

                current_solution = curSolution;

                //Where current solution comes from might need considering.
                /* if (option.current_solution_source == ParallelFusionOption::CURRENT_SOLUTION_FROM_SELF) */
                /*   current_solution = curSolution; */
                /* else if (option.current_solution_source == ParallelFusionOption::CURRENT_SOLUTION_FROM_BEST) { */
                /*   std::vector<std::pair<double, int> > solution_energy_index_pairs(slaveThreadIds.size()); */
                /*   for(auto tid=0; tid < slaveThreadIds.size(); ++tid) */
                /*     solution_energy_index_pairs[tid] = std::make_pair(bestSolutions[tid].getEnergy(), tid); */
                /*   std::vector<std::pair<double, int> >::const_iterator min_it = min_element(solution_energy_index_pairs.begin(), solution_energy_index_pairs.end()); */
                /*   int min_energy_thread_id = min_it->second; */
                /*   bestSolutions[min_energy_thread_id].get(current_solution); */
                /* } else if (option.current_solution_source == ParallelFusionOption::CURRENT_SOLUTION_FROM_RANDOM) { */
                /*   int selected_thread_id = rand() % slaveThreadIds.size(); */
                /*   bestSolutions[selected_thread_id].get(current_solution); */
                /* } */

                generator->writeSolution(curSolution, id, iter);

                //if synchronization is needed, thread won't submit solution unless
                //last submitted solution is read by the monitor thread.
                if(option.synchronize){
                    while(!write_flag[id].load())
                        std::this_thread::yield();
                }

                //set the current best solution. It will be visible from other threads
                bestSolutions[id].set(curSolution);

                if(option.synchronize)
                    write_flag[id].store(false);

                //double diffE =  lastEnergy - curSolution.first;
            }
        }catch(const std::exception& e){
            terminate.store(true);
            printf("Thread %d throws and exception: %s\n", id, e.what());
            return;
        }
    }

    template<class LABELSPACE>
    void ParallelFusionPipeline<LABELSPACE>::monitorThread(const int id, SolverPtr solver) {
        try{
            printf("Monitor thread launched\n");
            solver->initSolver(LABELSPACE());
            while(true) {
                if(terminate.load()) {
                    printf("Monitor thread quited\n");
                    break;
                }
                std::this_thread::yield();
                LABELSPACE proposals;
                SolutionType<LABELSPACE> current_solution;
                SolutionType<LABELSPACE> curSolution;
                for (auto tid = 0; tid < slaveThreadIds.size(); ++tid) {
                    if (option.synchronize) {
                        while (write_flag[tid].load())
                            std::this_thread::yield();
                    }
                    SolutionType<LABELSPACE> s;
                    bestSolutions[tid].get(s);
                    proposals.appendSpace(s.second);
                    if (option.synchronize)
                        write_flag[tid].store(true);
                }
                solver->solve(proposals, current_solution, curSolution);
            }
        }catch(const std::exception& e){
            terminate.store(true);
            printf("Thread %d throws and exception: %s\n", id, e.what());
            return;
        }
    }

    template<class LABELSPACE>
    void ParallelFusionPipeline<LABELSPACE>::getAllResult(LABELSPACE& solutions) const {
        for(auto i=0; i<slaveThreadIds.size(); ++i){
            SolutionType<LABELSPACE> s;
            bestSolutions[i].get(s);
            solutions.appendSpace(s.second);
        }
    }

}//namespace ParallelFusion
#endif
