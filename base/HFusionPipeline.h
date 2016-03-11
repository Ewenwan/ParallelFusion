//
// Created by yanhang on 3/9/16.
//

#ifndef PARALLELFUSION_HFUSIONPIPELINE_H
#define PARALLELFUSION_HFUSIONPIPELINE_H
#include "LabelSpace.h"
#include "FusionSolver.h"
#include "pipeline_util.h"
#include "thread_guard.h"
#include "binaryTree.h"

#include <opencv2/opencv.hpp>
#include <queue>
#include <future>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>

namespace ParallelFusion {
    template<class LABELSPACE>
    using SolutionPtr = std::shared_ptr<LABELSPACE>;

    template<class LABELSPACE>
    using NodeT = Hang_BinaryTree::Node<SolutionPtr<LABELSPACE> >;

    template<class LABELSPACE>
    using NodePtr = std::shared_ptr<NodeT<LABELSPACE> >;

    template<class LABELSPACE>
      struct AsynTask {
        AsynTask(NodePtr<LABELSPACE> p1, NodePtr<LABELSPACE> p2, NodeT<LABELSPACE> s) : lchild(p1), rchild(p2),
                                                                                        selfNode(s) { }

        NodePtr<LABELSPACE> lchild;
        NodePtr<LABELSPACE> rchild;
        NodeT<LABELSPACE> selfNode;
    };

    struct HFusionPipelineOption {
    public:
    HFusionPipelineOption() : num_threads(4) { }

        int num_threads;
    };

    template<class LABELSPACE>
    class HFusionPipeline {
    public:
        HFusionPipeline(const HFusionPipelineOption &option_) : option(option_), profile(new GlobalTimeEnergyProfile()),
                                                                finalSolution(new SynSolution<LABELSPACE>()) { }

        void runHFusion(const std::vector<LABELSPACE> &proposals,
                        std::vector<std::shared_ptr<FusionSolver<LABELSPACE> > > solver,
                        const LABELSPACE& initial);

        inline const SolutionType<LABELSPACE> &getSolution() const {
            return finalSolution->getSolution();
        }

    private:
        HFusionPipelineOption option;
        std::shared_ptr<GlobalTimeEnergyProfile> profile;
        std::shared_ptr<SynSolution<LABELSPACE> > finalSolution;
        Hang_BinaryTree::BinaryTree<SolutionPtr<LABELSPACE> > bTree;
    };


    template<class LABELSPACE>
    class ThreadPool {
    public:
        ThreadPool(const int num_threads,
                   const std::vector<std::shared_ptr<FusionSolver<LABELSPACE> > > solvers_,
                   const std::shared_ptr<GlobalTimeEnergyProfile> profile_,
                   const std::shared_ptr<SynSolution<LABELSPACE> > bestSolution_) :
                workers((size_t) num_threads), solvers(solvers_), profile(profile_), bestSolution(bestSolution_),
                timeout(100), terminate(false) {
            //launch threads. All threads will atomatically joined by thread_guard
            for (auto i = 0; i < workers.size(); ++i) {
                std::thread t(&ThreadPool::workerThread, this, i, solvers[i]);
                workers[i].bind(t);
            }
        }

        void submitTask(const NodePtr<LABELSPACE> t) {
            std::lock_guard<std::mutex> lock(mt);
            task_queue.push(t);
            cv.notify_all();
        }

        void workerThread(const int threadId, std::shared_ptr<FusionSolver<LABELSPACE> > solver);

        inline void stop() {
            terminate.store(true);
        }

    private:
        std::queue<NodePtr<LABELSPACE> > task_queue;
        std::vector<thread_guard> workers;

        std::vector<std::shared_ptr<FusionSolver<LABELSPACE> > > solvers;
        std::shared_ptr<GlobalTimeEnergyProfile> profile;
        std::shared_ptr<SynSolution<LABELSPACE> > bestSolution;
        mutable std::mutex mt;
        mutable std::condition_variable cv;
        const int timeout;
        float start_time;

        std::atomic<bool> terminate;
    };
    ///////////////////////////////////////////////////////////

    template<class LABELSPACE>
    void ThreadPool<LABELSPACE>::workerThread(const int threadId, std::shared_ptr<FusionSolver<LABELSPACE> > solver) {
        printf("Thread %d launched\n", threadId);
        while (!terminate.load()) {
            try {
                std::unique_lock<std::mutex> lock(mt);
                while (task_queue.empty()) {
                    cv.wait_for(lock, std::chrono::milliseconds(timeout));
                    if(terminate.load())
                        break;
                }
                if(terminate.load())
                    break;

                NodePtr<LABELSPACE> t = task_queue.front();
                task_queue.pop();
                lock.unlock();

                LABELSPACE p1 = *(t->lchild->data);
                p1.appendSpace(*(t->rchild->data));
                SolutionType<LABELSPACE> solution;
                SolutionType<LABELSPACE> current_solution;
                bestSolution->get(current_solution);
                solver->solve(p1, current_solution, solution);
                t->data = std::make_shared<LABELSPACE>(solution.second);

                //set flag1 to true, indicating this node is computed
                t->flag1.store(true);

                //record profile
                float difft = ((float) cv::getTickCount() - start_time) / (float) cv::getTickFrequency();
                Observation ob(difft, solution.first);
                profile->addObservation(ob);
                printf("Node: (%d,%d) fused by thread %d, final energy:%.5f\n", t->lchild->nodeId, t->rchild->nodeId, threadId, solution.first);
                //if current solution is better, update best soltuion
                if (ob.second < current_solution.first)
                    bestSolution->set(solution);

            } catch (const std::exception &e) {
                printf("Thread %d thrown an exception\n", threadId);
                terminate.store(true);
                break;
            }
        }
        printf("Thread %d quitted\n", threadId);
    }

    template<class LABELSPACE>
    void HFusionPipeline<LABELSPACE>::runHFusion(const std::vector<LABELSPACE> &proposals,
                                                 std::vector<std::shared_ptr<FusionSolver<LABELSPACE> > > solvers,
                                                 const LABELSPACE& initial) {
        CHECK_EQ(solvers.size(), option.num_threads) << "You must provide a thread-safe solver";
        //init solver
        for(auto& s: solvers)
            s->initSolver(initial);
        SolutionType<LABELSPACE> initSolution;
        initSolution.second = initial;
        initSolution.first = -1;
        finalSolution->set(initSolution);

        typedef typename Hang_BinaryTree::BinaryTree<SolutionPtr<LABELSPACE> >::FunctorType FunctorT;
        //build tree
        std::vector<SolutionPtr<LABELSPACE> > leaf(proposals.size());
        for (auto i = 0; i < leaf.size(); ++i) {
            leaf[i] = SolutionPtr<LABELSPACE>(new LABELSPACE(proposals[i]));
        }
        bTree.buildFromLeafData(leaf);
        //set all leaf to be ready.
        //In HFusion, we use two auxiliary flags in Node structure. flag1 indicate whether the solution is ready,
        //flag2 indicates whether the fusion has already been submitted
        FunctorT fSetLeaf = [&](NodePtr<LABELSPACE> node) {
            if (node->lchild.get() || node->rchild.get())
                return;
            node->flag1.store(true);
        };
        bTree.traverseTree(fSetLeaf, Hang_BinaryTree::POST_ORDER, bTree.getRoot());

        //initialize thread pool
        ThreadPool<LABELSPACE> threadPool(option.num_threads, solvers, profile, finalSolution);
        //repeately traverse the tree,
        //traverse function
        FunctorT fFusion = [&](NodePtr<LABELSPACE> node) {
            //if node has two children, and their result are both ready, and hasn't been submitted to the thread poll,
            //submit a fusion task to the thread pool
            if (node->lchild.get() && node->rchild.get() && (!node->flag2.load())) {
                if (node->lchild->flag1.load() && node->rchild->flag1.load()) {
                    node->flag2.store(true);
                    threadPool.submitTask(node);
                }
            }
        };

        //loop until the solution of root node is ready
        while (!bTree.getRoot()->flag1.load()) {
            bTree.traverseTree(fFusion, Hang_BinaryTree::POST_ORDER, bTree.getRoot());
        }
        printf("All done\n");
        //stop thread pool
        threadPool.stop();
        printf("thread pool stopped\n");
    }
}//namespace ParallelFusion

#endif //PARALLELFUSION_HFUSIONPIPELINE_H
