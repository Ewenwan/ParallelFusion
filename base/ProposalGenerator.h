#ifndef PROPOSAL_GENERATOR_H__
#define PROPOSAL_GENERATOR_H__

#include "LabelSpace.h"
#include <memory>
#include <vector>

namespace ParallelFusion {

// Note: a single instance of proposal generator might be accessed from multiple
// threads. If getProposals() changes
// internal data, please provide necessary synchronization
template <class LABELSPACE> class ProposalGenerator {
public:
  // method of generating proposals.
  // input: current_solution
  // output: proposal
  // NOTE: the method only return one proposal one time!!!!!
  // If you need multiple proposals, call this method multiple times. The
  // concrete subclass should know which
  // proposal to generate (e.g. by a internal counter). Again, be careful of
  // synchronization
  virtual void getProposals(LABELSPACE &proposals,
                            const LABELSPACE &current_solution,
                            const int N) = 0;

  virtual void
  writeSolution(const std::pair<double, LABELSPACE> &solution,
                const int thread_index, const int iteration,
                const std::vector<int> &selected_threads){}; // write temporary
                                                             // solution for
                                                             // visualization
                                                             // purpose
  // virtual void setGroundTruthSolution(const std::vector<LabelType> &solution)
  // {}; //Set ground truth solution for debugging purpose. Better to be set in
  // child classes' constructors.
};
}
#endif
