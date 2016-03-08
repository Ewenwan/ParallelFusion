#ifndef FUSION_SOLVER_H__
#define FUSION_SOLVER_H__

#include <vector>

#include "LabelSpace.h"

namespace ParallelFusion {

    //FusionSolver class represents both energy definition and solving procedure.
    template<typename LabelType>
    class FusionSolver {
    public:
        //take initial value, run optimization and store solution in 'solution', return error
        virtual double solve(const LabelSpace<LabelType> &proposals, std::vector<LabelType> &solution) const = 0;

        //initialize solver with an initial solution, will be called automatically
        virtual void initSolver(const std::vector<LabelType>& initial){}

        //given a solution, evaluate energy
        virtual double evaluateEnergy(const std::vector<LabelType>& solution) const = 0;
    };
}//namespace ParallelFusion
#endif
