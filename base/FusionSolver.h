#ifndef FUSION_SOLVER_H__
#define FUSION_SOLVER_H__

#include <vector>

#include "LabelSpace.h"

namespace ParallelFusion {

    //FusionSolver class represents both energy definition and solving procedure.
    template<class LABELSPACE>
    class FusionSolver {
    public:
        //take initial value, run optimization and store solution in 'solution', return error
        virtual double solve(const LABELSPACE &proposals, LABELSPACE &solution) const = 0;

        //initialize solver with an initial solution, will be called automatically
        virtual void initSolver(const LABELSPACE & initial){}

        //given a solution, evaluate energy
	virtual double evaluateEnergy(const LABELSPACE & solution) const = 0;
    };
}//namespace ParallelFusion
#endif
