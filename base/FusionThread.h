#ifndef FUSION_THREAD_H__
#define FUSION_THREAD_H__

#include <vector>
#include <memory>

#include "ProposalGenerator.h"
#include "FusionSolver.h"
#include "cv_utils/cv_utils.h"

namespace ParallelFusion {
    template<class LabelType>
    class FusionThread {
    public:
        FusionThread(const std::shared_ptr<ProposalGenerator<LabelType> > &proposal_generator,
                     const std::shared_ptr<FusionSolver<LabelType> > &fusion_solver,
                     const double MASTER_LIKELIHOOD) :
                proposal_generator_(proposal_generator), fusion_solver_(fusion_solver),
                MASTER_LIKELIHOOD_(MASTER_LIKELIHOOD) {
            updateStatus();
        }

        //two parameters are just for debugging purpose
        void runFusion(const int thread_index = 0,
                       const int iteration = 0) {
            if (is_master_) {
                if (fused_solution_.size() > 0)
                    solution_collection_ += LabelSpace<LabelType>(fused_solution_);
                std::vector<std::vector<LabelType> > solution_collection_vec = solution_collection_.getLabelSpace();
                fused_solution_energy_ = fusion_solver_->solve(solution_collection_, fused_solution_);
            } else {
                proposal_generator_->setCurrentSolution(current_solution_);
                LabelSpace<LabelType> label_space;
                proposal_generator_->getProposal(label_space);
                fused_solution_energy_ = fusion_solver_->solve(solution_collection_, fused_solution_);
            }

            proposal_generator_->writeSolution(fused_solution_, thread_index,
                                               iteration); //write solution to files for debugging
            updateStatus();
        }

        void setCurrentSolution(
                const std::vector<LabelType> &current_solution) { current_solution_ = current_solution; };

        std::vector<LabelType> getFusedSolution() { return fused_solution_; };

        double getFusedSolutionEnergy() { return fused_solution_energy_; };

        void setSolutionCollection(
                const LabelSpace<LabelType> &solution_collection) { if (is_master_) solution_collection_ = solution_collection; }; //Solution_collection is a label space.
        void setSolutionPool(
                const std::vector<std::vector<LabelType> > &solution_pool) //Solution_pool is a set of solutions.
        {
            if (is_master_ && solution_pool.size() > 0) {
                solution_collection_.assign(solution_pool.begin()->size());
                for (typename std::vector<std::vector<LabelType> >::const_iterator solution_it = solution_pool.begin();
                     solution_it != solution_pool.end(); solution_it++)
                    solution_collection_ += LabelSpace<LabelType>(*solution_it);
            }
        }

        int getStatus() const { return is_master_; };

    private:
        std::vector<LabelType> current_solution_;
        std::vector<LabelType> fused_solution_;
        double fused_solution_energy_;

        const double MASTER_LIKELIHOOD_;
        bool is_master_;

        LabelSpace<LabelType> solution_collection_;
        const std::shared_ptr<ProposalGenerator<LabelType> > &proposal_generator_;
        const std::shared_ptr<FusionSolver<LabelType> > &fusion_solver_;

        void updateStatus() { is_master_ = cv_utils::randomProbability() < MASTER_LIKELIHOOD_; };
    };
}//namespace ParallelFusion
#endif
