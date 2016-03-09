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
#include <mutex>
#include "LabelSpace.h"
#include "ProposalGenerator.h"


namespace ParallelFusion {

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

    struct ParallelFusionOption {
    ParallelFusionOption() : convergeThreshold(0.01), max_iteration(10), num_threads(6), addMethod(APPEND) { }
        //Addition method: how add two proposals. APPEND: simply append; UNION: take union, remove duplicate labels
        enum ProposalAddition{APPEND, UNION};
      double convergeThreshold;
        int max_iteration;
        int num_threads;
      ProposalAddition addMethod;
    };

    struct ThreadOption {
    ThreadOption() : kSelfThread(2), kOtherThread(0) { }
      int kSelfThread;
      int kOtherThread;   
    };
    
    template<class LABELSPACE>
    class ParallelFusionPipeline {
    public:
        //solver should be read only
        ParallelFusionPipeline(const ParallelFusionOption &option_) : option(option_), terminate(false) { }

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

        //slave threads
        void workerThread(const int id,
                          const LABELSPACE& initial,
                          const GeneratorPtr &generator,
                          const SolverPtr& solver,
			  const ThreadOption &thread_option);

        inline const ParallelFusionOption &getOption() const { return option; }

        inline ParallelFusionOption &getOption() { return option; }


        ParallelFusionPipeline(ParallelFusionPipeline &) = delete;

        ParallelFusionPipeline &operator=(const ParallelFusionPipeline &) = delete;

    private:
        //store current best solutions from each thread. The array can be access by multiples threads
        //each solution store a vector of labeling, and corresponding energy
        std::vector<std::shared_ptr<SynSolution<LABELSPACE> > > bestSolutions;

        //The following two parameters controls the parallel fusion.
        //each fusion
        ParallelFusionOption option;

        //if one thread throws exception, notify all threads
        std::atomic<bool> terminate;
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

        bestSolutions.resize((size_t)option.num_threads);
	for(auto i=0; i<bestSolutions.size(); ++i){
            bestSolutions[i] = std::shared_ptr<SynSolution<LABELSPACE> >(
                    new SynSolution<LABELSPACE>()
            );
            bestSolutions[i]->getSolution().second = initials[i];
        }

	for(auto i=0; i<bestSolutions.size(); ++i){
	  CHECK_GE(thread_options[i].kSelfThread, 0) << "Negative number of proposals from self.";
	  CHECK_GE(thread_options[i].kOtherThread, 0) << "Negative number of proposals from others.";
	}

        //launch threads. Join method is called from the destructor of thread_guard
        std::vector<thread_guard> slaves(option.num_threads);
        for(auto tid=0; tid<slaves.size(); ++tid){
	  printf("Lauching threads %d...\n", tid);
	  std::thread t(&ParallelFusionPipeline::workerThread, this, tid, std::ref(initials[tid]), std::ref(generators[tid]), std::ref(solvers[tid]), std::ref(thread_options[tid]));
            slaves[tid].bind(t);
        }
    }

    template<class LABELSPACE>
    double ParallelFusionPipeline<LABELSPACE>::getBestLabeling(SolutionType<LABELSPACE> &solution) const {
        double minE = std::numeric_limits<double>::max();
        for(auto i=0; i<bestSolutions.size(); ++i){
            SolutionType<LABELSPACE> cursolution;
            bestSolutions[i]->get(cursolution);
            if(cursolution.first < minE){
                solution.second.clear();
                solution.second.appendSpace(cursolution.second);
                solution.first = cursolution.first;
                minE = cursolution.first;
            }
        }
        return minE;
    }

    template<class LABELSPACE>
      void ParallelFusionPipeline<LABELSPACE>::workerThread(const int id,
							    const LABELSPACE& initial,
                                                 const GeneratorPtr& generator,
							    const SolverPtr& solver,
							    const ThreadOption &thread_option){
        try {
            printf("Thread %d lauched\n", id);
            std::default_random_engine seed;
            std::uniform_int_distribution<int> distribution(0, option.num_threads-1);
            bool converge = false;

            solver->initSolver(initial);

            SolutionType<LABELSPACE> current_solution;
            current_solution.first = solver->evaluateEnergy(initial);
            current_solution.second.appendSpace(initial);
            bestSolutions[id]->set(current_solution);

            double lastEnergy = current_solution.first;

            for (auto iter = 0; iter < option.max_iteration; ++iter) {
                if(terminate.load()){
                    printf("Thread %d quited\n", id);
                    return;
                }
                LABELSPACE proposals;
                //generate proposal by own generator
                LABELSPACE proposals_self;
                printf("Generating proposals...\n");
		
                generator->getProposals(proposals_self, current_solution.second, thread_option.kSelfThread);
		
                if(option.addMethod == ParallelFusionOption::APPEND)
                    proposals.appendSpace(proposals_self);
                else
                    proposals.unionSpace(proposals_self);
                //Take best solutions from other threads. Initially there is no 'best solution', marked by
                //the energy less than 0. If such condition occurs, replace this with another self generated
                //proposal.
                for(auto pid=0; pid < thread_option.kOtherThread; ++pid) {
                    int tid = distribution(seed);
                    SolutionType<LABELSPACE> s;
                    bestSolutions[tid]->get(s);
                    if(s.first >= 0) {
                        if (option.addMethod == ParallelFusionOption::APPEND)
                            proposals.appendSpace(s.second);
                        else
                            proposals.unionSpace(s.second);
                    }else{
                        LABELSPACE tempProposal;
                        generator->getProposals(tempProposal, current_solution.second, 1);
                        if(option.addMethod == ParallelFusionOption::APPEND)
                            proposals.appendSpace(tempProposal);
                        else
                            proposals.unionSpace(tempProposal);
                    }
                }

                //solve
                SolutionType<LABELSPACE> curSolution;
                printf("Solving...\n");
                solver->solve(proposals, current_solution, curSolution);
                printf("Done. Energy: %.5f\n", curSolution.first);
                current_solution = curSolution;
		generator->writeSolution(curSolution, id, iter);

                //set the current best solution. It will be visible from other threads
                bestSolutions[id]->set(curSolution);
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
