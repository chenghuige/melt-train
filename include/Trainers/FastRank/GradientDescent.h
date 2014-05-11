/**
 *  ==============================================================================
 *
 *          \file   GradientDescent.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 15:11:19.761467
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef GRADIENT_DESCENT_H_
#define GRADIENT_DESCENT_H_
#include "common_def.h"
#include "OptimizationAlgorithm.h"
namespace gezi {

class GradientDescent : public OptimizationAlgorithm
{
public:
	virtual RegressionTree TrainingIteration() override
	{
		RegressionTree tree = TreeLearner.FitTargets(AdjustTargetsAndSetWeights());
		return tree;
	}

	virtual dvec AdjustTargetsAndSetWeights()
	{
		dvec targets;
		return targets;
	}

	virtual dvec& GetGradient()
	{
		return ObjectiveFunction.GetGradient(TrainingScores->Scores);
	}

	virtual ScoreTrackerPtr ConstructScoreTracker(string name, Dataset set, dvec& initScores) override
	{
		return make_shared<ScoreTracker>(name, set, initScores);
	}


protected:
private:

};

}  //----end of namespace gezi

#endif  //----end of GRADIENT_DESCENT_H_
