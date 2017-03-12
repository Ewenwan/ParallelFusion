#ifndef PARALLEL_FUSION_PIPELINE_H__
#define PARALLEL_FUSION_PIPELINE_H__

#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <random>
#include <vector>

//#include "FusionThread.h"
//#include "cv_utils/cv_utils.h"
#include "pipeline_util.h"

using namespace fmt::literals;

namespace ParallelFusion {

template <class LABELSPACE> class ParallelFusionPipeline {
public:
  // solver should be read only
  ParallelFusionPipeline(const ParallelFusionOption &option_)
      : option(option_), bestSolutions((size_t)option_.num_threads),
        terminate(false), write_flag((size_t)option_.num_threads),
        threadProfile((size_t)option_.num_threads) {
    start_time = (float)cv::getTickCount();
    final_fused.first = std::numeric_limits<double>::max();
  }

  typedef std::shared_ptr<ProposalGenerator<LABELSPACE>> GeneratorPtr;
  typedef std::vector<GeneratorPtr> GeneratorSet;
  typedef std::shared_ptr<FusionSolver<LABELSPACE>> SolverPtr;
  typedef std::vector<SolverPtr> SolverSet;

  // run parallel fusion. The logic in this routine represents the master thread
  // input: num_threads: number of threads to use
  //       max_iter: maximum iterations of fusion in each thread
  //       initials: initial solution for each thread. The size must be
  //       num_threads
  //       gnerators: shared pointers of ProposalGenerator, one for each thread
  //       solvers: shared pointers of FusionSolver, one for each thread
  // return: final energy
  double runParallelFusion(const std::vector<LABELSPACE> &initials,
                           const GeneratorSet &generators,
                           const SolverSet &solvers,
                           const std::vector<ThreadOption> &thread_options,
                           const bool reset_time = true);

  double getBestLabeling(SolutionType<LABELSPACE> &solution) const;
  void getAllResult(LABELSPACE &solutions) const;
  inline const GlobalTimeEnergyProfile &getGlobalProfile() const {
    return globalProfile;
  }
  inline GlobalTimeEnergyProfile &getGlobalProfile() { return globalProfile; }
  inline const std::vector<std::list<Observation>> &
  getAllThreadProfiles() const {
    return threadProfile;
  }
  void final_fusion(SolverPtr solver);

  // slave threads
  virtual void workerThread(const int id, const LABELSPACE &initial,
                            GeneratorPtr generator, SolverPtr solver,
                            const ThreadOption &thread_option);
  virtual void monitorThread(const int id, GeneratorPtr generator,
                             SolverPtr solver,
                             const ThreadOption &thread_option);

  inline const ParallelFusionOption &getOption() const { return option; }

  inline ParallelFusionOption &getOption() { return option; }

  ParallelFusionPipeline(ParallelFusionPipeline &) = delete;

  ParallelFusionPipeline &operator=(const ParallelFusionPipeline &) = delete;

protected:
  // store current best solutions from each thread. The array can be access by
  // multiples threads
  // each solution store a vector of labeling, and corresponding energy
  std::vector<SynSolution<LABELSPACE>> bestSolutions;

  ParallelFusionOption option;
  // if one thread throws exception, notify all threads
  std::atomic<bool> terminate;
  // In the presence of monitor thread, all threads might need to be
  // synchronized.
  std::vector<std::atomic<bool>> write_flag;

  std::vector<int> slaveThreadIds;
  std::vector<int> monitorThreadIds;

  // keep tracking of <time, energy> of each thread
  std::vector<std::list<Observation>> threadProfile;

  // global time-energy profile
  GlobalTimeEnergyProfile globalProfile;

  float start_time;
  SolutionType<LABELSPACE> final_fused;
};

///////////////////////////////////////////////////////////////////////

template <class LABELSPACE>
double ParallelFusionPipeline<LABELSPACE>::runParallelFusion(
    const std::vector<LABELSPACE> &initials, const GeneratorSet &generators,
    const SolverSet &solvers, const std::vector<ThreadOption> &thread_options,
    const bool reset_time) {

  CHECK_EQ(option.num_threads, initials.size());
  CHECK_EQ(option.num_threads, generators.size());
  CHECK_EQ(option.num_threads, solvers.size());
  CHECK_EQ(option.num_threads, thread_options.size());

  bool monitor_exists = false;

  monitorThreadIds.clear();
  slaveThreadIds.clear();
  terminate.store(false);

  // sanity checks
  for (auto i = 0; i < option.num_threads; ++i) {
    if (thread_options[i].is_monitor) {
      monitor_exists = true;
      // CHECK_EQ(thread_options[i].kTotal, thread_options[i].kOtherThread) <<
      // "Monitor thread can not generate proposal";
      monitorThreadIds.push_back(i);
    } else {
      slaveThreadIds.push_back(i);
    }
    CHECK_GE(thread_options[i].kTotal, 0)
        << "Negative number of proposals from self.";
    CHECK_GE(thread_options[i].kOtherThread, 0)
        << "Negative number of proposals from others.";
    CHECK_GE(thread_options[i].solution_exchange_interval, 1)
        << "Solution exchange interval should be a positive number.";
  }
  // if synchronization is needed, there must be an monitor thread
  if (option.synchronize)
    CHECK(monitor_exists);

  // initialize arrays. Slave threads are store in 0..slaveThreadIds.size()
  // elements
  for (auto i = 0; i < slaveThreadIds.size(); ++i) {
    const int &idx = slaveThreadIds[i];
    bestSolutions[i].set(SolutionType<LABELSPACE>(-1, initials[idx]));
    write_flag[i].store(true);
  }

  if (reset_time)
    start_time = (float)cv::getTickCount();

  // launch slave threads. Join method is called from the destructor of
  // thread_guard
  std::vector<thread_guard> slaves(slaveThreadIds.size());
  for (auto tid = 0; tid < slaves.size(); ++tid) {
    printf("Lauching slave threads %d...\n", tid);
    const int &idx = slaveThreadIds[tid];
    std::thread t(&ParallelFusionPipeline::workerThread, this, tid,
                  std::ref(initials[idx]), generators[idx], solvers[idx],
                  thread_options[idx]);
    slaves[tid].bind(t);
  }

  // launch monitor threads
  std::vector<thread_guard> monitors(monitorThreadIds.size());
  for (auto tid = 0; tid < monitorThreadIds.size(); ++tid) {
    printf("Lauching monitor threads %d...\n", tid);
    const int &idx = monitorThreadIds[tid];
    std::thread t(&ParallelFusionPipeline::monitorThread, this, tid,
                  generators[idx], solvers[idx], thread_options[idx]);
    monitors[tid].bind(t);
  }

  // wait for all slave threads to finish job
  for (auto i = 0; i < slaves.size(); ++i)
    slaves[i].join();

  // quit monitor thread
  terminate.store(true);

  if (thread_options[slaveThreadIds[0]].kOtherThread == 0)
    final_fusion(solvers[0]);
}

template <class LABELSPACE>
double ParallelFusionPipeline<LABELSPACE>::getBestLabeling(
    SolutionType<LABELSPACE> &solution) const {
  double minE = std::numeric_limits<double>::max();
  for (auto i = 0; i < slaveThreadIds.size(); ++i) {
    if (bestSolutions[i].getSolution().first < minE) {
      solution.second.clear();
      solution.second.appendSpace(bestSolutions[i].getSolution().second);
      solution.first = bestSolutions[i].getSolution().first;
      minE = solution.first;
    }
  }

  if (final_fused.first < minE) {
    solution.second.clear();
    solution.second.appendSpace(final_fused.second);
    solution.first = final_fused.first;
    minE = solution.first;
  }

  return minE;
}

template <class LABELSPACE>
void ParallelFusionPipeline<LABELSPACE>::workerThread(
    const int id, const LABELSPACE &initial, GeneratorPtr generator,
    SolverPtr solver, const ThreadOption &thread_option) {

  try {
    fmt::print("Thread {} launched\n", id);
    std::seed_seq seed{id, 1, 2, 4, id, 5, 6, id, 7, 8, id};
    std::mt19937_64 rng(seed);
    bool converge = false;

    solver->initSolver(initial);

    SolutionType<LABELSPACE> current_solution;
    current_solution.first = solver->evaluateEnergy(initial);
    current_solution.second = initial;
    bestSolutions[id].set(current_solution);

    double lastEnergy = current_solution.first;
    double initTime = ((float)cv::getTickCount() - start_time) /
                      (float)cv::getTickFrequency();
    globalProfile.addObservation(initTime, current_solution.first);
    threadProfile[slaveThreadIds[id]].push_back(
        Observation(initTime, current_solution.first));

    std::chrono::nanoseconds run_time(0);
    for (int iter = 0; iter < option.max_iteration; ++iter) {
      if (terminate.load()) {
        fmt::print("Thread {} quited\n", id);
        return;
      }
      auto start = std::chrono::system_clock::now();
      LABELSPACE proposals;
      // generate proposal by own generator
      LABELSPACE proposals_self;
      // printf("Generating proposals...\n");
      const int num_proposals_from_others =
          (iter + 1) % thread_option.solution_exchange_interval == 0
              ? thread_option.kOtherThread
              : 0;

      bool grabbed_solution_from_self = false;
      std::vector<int> selected_threads;
      if (num_proposals_from_others > 0) {
        LOG(INFO) << "Thread: {}\tGetting {} from other threads"_format(
            id, num_proposals_from_others);
        if (option.selectionMethod == ParallelFusionOption::RANDOM) {
          std::vector<int> all_other_worker_threads;
          for (auto tid = 0; tid < slaveThreadIds.size(); ++tid)
            if (tid != id)
              all_other_worker_threads.push_back(tid);

          std::shuffle(all_other_worker_threads.begin(),
                       all_other_worker_threads.end(), rng);

          for (auto pid = 0; pid < num_proposals_from_others; ++pid) {
            int tid = all_other_worker_threads[pid];
            SolutionType<LABELSPACE> s;
            bestSolutions[tid].get(s);
            proposals.appendSpace(s.second);

            selected_threads.push_back(slaveThreadIds[tid]);
          }

        } else if (option.selectionMethod == ParallelFusionOption::BEST) { //

          std::vector<std::pair<double, int>> solution_energy_index_pairs;

          for (auto tid = 0; tid < slaveThreadIds.size(); ++tid)
            if (tid != id)
              solution_energy_index_pairs.push_back(
                  std::make_pair(bestSolutions[tid].getEnergy(), tid));

          sort(solution_energy_index_pairs.begin(),
               solution_energy_index_pairs.end());

          for (auto pid = 0; pid < num_proposals_from_others; ++pid) {

            int tid = solution_energy_index_pairs[pid].second;
            if (tid == id) {
              grabbed_solution_from_self = true;
              continue;
            }

            SolutionType<LABELSPACE> s;
            bestSolutions[tid].get(s);
            proposals.appendSpace(s.second);

            selected_threads.push_back(slaveThreadIds[tid]);
          }

        } else if (option.selectionMethod == ParallelFusionOption::ALL) { //

          proposals.appendSpace(current_solution.second);
          int num_proposals_to_fuse = 1;

          for (auto tid = 0; tid < slaveThreadIds.size(); ++tid) {
            if (tid == id)
              continue;

            SolutionType<LABELSPACE> s;
            bestSolutions[tid].get(s);
            proposals.appendSpace(s.second);

            selected_threads.push_back(slaveThreadIds[tid]);

            num_proposals_to_fuse++;
            if (num_proposals_to_fuse == thread_option.kTotal + 1) {
              SolutionType<LABELSPACE> curSolution;
              solver->solve(proposals, current_solution, curSolution);
              current_solution = curSolution;
              proposals = curSolution.second;
              num_proposals_to_fuse = 1;
            }
          }

          //                        float dt = ((float)cv::getTickCount() -
          //                        start_time) / (float)cv::getTickFrequency();
          //                        threadProfile[slaveThreadIds[id]].push_back(Observation(dt,
          //                        current_solution.first));
          //                        globalProfile.addObservation(dt,
          //                        current_solution.first);

          generator->writeSolution(current_solution, slaveThreadIds[id], iter,
                                   selected_threads);

          // if synchronization is needed, thread won't submit solution unless
          // last submitted solution is read by the monitor thread.
          if (option.synchronize) {
            while (!write_flag[id].load())
              std::this_thread::yield();
          }

          // bestSolutions[id].set(current_solution);
          // if(option.synchronize) //Always set this flag so that a minotor
          // could know whether results are ready, synchronization is guaranteed
          // it the minotor thread set the flag to true.
          write_flag[id].store(false);

          continue;
        }
      }

      int num_proposals_from_self =
          thread_option.kTotal - num_proposals_from_others;
      if (grabbed_solution_from_self)
        num_proposals_from_self++;
      generator->getProposals(proposals_self, current_solution.second,
                              num_proposals_from_self);
      proposals.appendSpace(proposals_self);

      // printf("In iteration %d, thread %d generates %d proposals and grab %d
      //        solutions\n ", iter, id, num_proposals_from_self,
      //        num_proposals_from_others);

      // std::cout << "Thread " << id << " generates " <<
      // num_proposals_from_self
      //           << " proposals and grabs " << num_proposals_from_others
      //           << " solutions from others" << std::endl;

      // printf("Solving...\n");
      SolutionType<LABELSPACE> curSolution;
      solver->solve(proposals, current_solution, curSolution);
      LOG(INFO) << "Thread: {}\tcurrent energy: {}"_format(id,
                                                           curSolution.first);

      // write thread profile, update global profile
      float dt = ((float)cv::getTickCount() - start_time) /
                 (float)cv::getTickFrequency();
      threadProfile[slaveThreadIds[id]].push_back(
          Observation(dt, curSolution.first));
      globalProfile.addObservation(dt, curSolution.first);

      current_solution = curSolution;

      // Where current solution comes from might need considering.
      /* if (option.current_solution_source ==
       * ParallelFusionOption::CURRENT_SOLUTION_FROM_SELF) */
      /*   current_solution = curSolution; */
      /* else if (option.current_solution_source ==
       * ParallelFusionOption::CURRENT_SOLUTION_FROM_BEST) { */
      /*   std::vector<std::pair<double, int> >
       * solution_energy_index_pairs(slaveThreadIds.size()); */
      /*   for(auto tid=0; tid < slaveThreadIds.size(); ++tid) */
      /*     solution_energy_index_pairs[tid] =
       * std::make_pair(bestSolutions[tid].getEnergy(), tid); */
      /*   std::vector<std::pair<double, int> >::const_iterator min_it =
       * min_element(solution_energy_index_pairs.begin(),
       * solution_energy_index_pairs.end()); */
      /*   int min_energy_thread_id = min_it->second; */
      /*   bestSolutions[min_energy_thread_id].get(current_solution); */
      /* } else if (option.current_solution_source ==
       * ParallelFusionOption::CURRENT_SOLUTION_FROM_RANDOM) { */
      /*   int selected_thread_id = rand() % slaveThreadIds.size(); */
      /*   bestSolutions[selected_thread_id].get(current_solution); */
      /* } */

      generator->writeSolution(curSolution, slaveThreadIds[id], iter,
                               selected_threads);

      // if synchronization is needed, thread won't submit solution unless
      // last submitted solution is read by the monitor thread.
      if (option.synchronize) {
        while (!write_flag[id].load())
          std::this_thread::yield();
      }

      // set the current best solution. It will be visible from other threads
      bestSolutions[id].set(curSolution);

      // if(option.synchronize) //Always set this flag so that a minotor could
      // know whether results are ready, synchronization is guaranteed it the
      // minotor thread set the flag to true.
      write_flag[id].store(false);

      // double diffE =  lastEnergy - curSolution.first;

      auto end = std::chrono::system_clock::now();
      std::chrono::nanoseconds current_run = end - start;
      run_time += current_run;

      if (run_time + current_run >= option.timeout) {
        fmt::print("Thread: {}\tRan for: {}\n", id, run_time.count() / 1e9);
        terminate.store(true);
      }
    }
  } catch (const std::exception &e) {
    terminate.store(true);
    printf("Thread %d throws and exception: %s\n", id, e.what());
    return;
  }
}

template <class LABELSPACE>
void ParallelFusionPipeline<LABELSPACE>::monitorThread(
    const int id, GeneratorPtr generator, SolverPtr solver,
    const ThreadOption &thread_option) {
  try {
    printf("Monitor thread launched\n");
    solver->initSolver(LABELSPACE());
    int iter = 0;

    while (true) {
      if (terminate.load()) {
        printf("Monitor thread quited\n");
        break;
      }

      std::this_thread::yield();
      LABELSPACE proposals;
      SolutionType<LABELSPACE> current_solution;

      current_solution.first = std::numeric_limits<double>::max();
      CHECK_LE(thread_option.kTotal, slaveThreadIds.size())
          << "Not enough slave threads for final fusion.";

      int num_proposals_to_fuse = 0;
      for (auto tid = 0; tid < slaveThreadIds.size(); ++tid) {
        // if (option.synchronize) No matter synchronize or not, monitor thread
        // has to wait util results are available.
        while (write_flag[tid].load())
          std::this_thread::yield();

        SolutionType<LABELSPACE> s;
        bestSolutions[tid].get(s);
        proposals.appendSpace(s.second);
        if (option.synchronize)
          write_flag[tid].store(true);

        num_proposals_to_fuse++;
        if (num_proposals_to_fuse == thread_option.kTotal + 1) {
          std::cout << "Monitor fusing" << std::endl;
          SolutionType<LABELSPACE> curSolution;
          solver->solve(proposals, current_solution, curSolution);
          current_solution = curSolution;
          proposals = curSolution.second;
          num_proposals_to_fuse = 1;
        }
      }

      float dt = ((float)cv::getTickCount() - start_time) /
                 static_cast<float>(cv::getTickFrequency());
      threadProfile[monitorThreadIds[id]].push_back(
          Observation(dt, current_solution.first));
      globalProfile.addObservation(dt, current_solution.first);

      generator->writeSolution(current_solution, monitorThreadIds[id], iter,
                               std::vector<int>());
      iter++;
    }
  } catch (const std::exception &e) {
    terminate.store(true);
    printf("Thread %d throws and exception: %s\n", id, e.what());
    return;
  }
}

template <class LABELSPACE>
void ParallelFusionPipeline<LABELSPACE>::getAllResult(
    LABELSPACE &solutions) const {
  for (auto i = 0; i < slaveThreadIds.size(); ++i) {
    SolutionType<LABELSPACE> s;
    bestSolutions[i].get(s);
    solutions.appendSpace(s.second);
  }
}

template <class LABELSPACE>
void ParallelFusionPipeline<LABELSPACE>::final_fusion(SolverPtr solver) {
  fmt::print("Performing final fusion!\n");
  LABELSPACE proposals;

  final_fused.first = std::numeric_limits<double>::max();

  int num_proposals_to_fuse = 0;
  for (auto tid = 0; tid < slaveThreadIds.size(); ++tid) {
    // if (option.synchronize) No matter synchronize or not, monitor thread
    // has to wait util results are available.

    SolutionType<LABELSPACE> s;
    bestSolutions[tid].get(s);
    proposals.appendSpace(s.second);

    num_proposals_to_fuse++;
    if (num_proposals_to_fuse == 2) {
      fmt::print("Fusing!\n");
      SolutionType<LABELSPACE> curSolution;
      solver->solve(proposals, final_fused, curSolution);
      final_fused = curSolution;
      proposals = curSolution.second;
      num_proposals_to_fuse = 1;
    }
  }
}

} // namespace ParallelFusion
#endif
