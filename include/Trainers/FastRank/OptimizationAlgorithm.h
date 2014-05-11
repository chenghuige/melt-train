/**
 *  ==============================================================================
 *
 *          \file   OptimizationAlgorithm.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 14:35:32.002966
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef OPTIMIZATION_ALGORITHM_H_
#define OPTIMIZATION_ALGORITHM_H_
#include "common_def.h"
#include "TreeLearner.h"
#include "RegressionTree.h"
#include "ObjectiveFunction.h"
#include "ScoreTracker.h"
#include "Ensemble.h"
namespace gezi {

class OptimizationAlgorithm 
{
public:
	virtual RegressionTree TrainingIteration() = 0;

	OptimizationAlgorithm(::Ensemble& ensemble, Dataset& trainData, dvec& initTrainScores)
		:Ensemble(ensemble)
	{
		TrainingScores = ConstructScoreTracker("train", trainData, initTrainScores);
		TrackedScores.push_back(TrainingScores);
	}

public:
	shared_ptr<TreeLearner> TreeLearner; //@TODO unique_ptr就可以了？
	shared_ptr<ObjectiveFunction> ObjectiveFunction;
	Ensemble& Ensemble;
	double Smoothing;
	vector<ScoreTracker> TrackedScores;
	ScoreTracker TrainingScores;
	bool useFastTrainingScoresUpdate = true;
};

}  //----end of namespace gezi

#endif  //----end of OPTIMIZATION_ALGORITHM_H_
