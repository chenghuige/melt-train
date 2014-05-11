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
#include "IStepSearch.h"
namespace gezi {

class OptimizationAlgorithm 
{
public:
	OptimizationAlgorithm(::Ensemble& ensemble, Dataset& trainData, dvec& initTrainScores)
		:Ensemble(ensemble)
	{
		TrainingScores = ConstructScoreTracker("train", trainData, initTrainScores);
		TrackedScores.push_back(TrainingScores);
	}

	virtual RegressionTree& TrainingIteration() = 0;
	virtual ScoreTrackerPtr ConstructScoreTracker(string name, Dataset set, dvec& InitScores) = 0;

	virtual void FinalizeLearning(int bestIteration)
	{
		if (bestIteration != Ensemble.NumTrees())
		{
			Ensemble.RemoveAfter(std::max(bestIteration, 0));
			TrackedScores.clear();
		}
	}

	ScoreTrackerPtr GetScoreTracker(string name, Dataset& set, dvec& InitScores)
	{
		for(ScoreTrackerPtr st : TrackedScores)
		{
			if (&(st->Dataset) == &set)
			{
				return st;
			}
		}
		ScoreTrackerPtr newTracker = ConstructScoreTracker(name, set, InitScores);
		TrackedScores.push_back(newTracker);
		return newTracker;
	}

	void SetTrainingData(Dataset& trainData, dvec& initTrainScores)
	{
		TrainingScores = ConstructScoreTracker("train", trainData, initTrainScores);
		TrackedScores[0] = TrainingScores;
	}

	virtual void SmoothTree(RegressionTree& tree, double smoothing)
	{
		//if (smoothing != 0.0)
		{//@TODO smooth
			
		}
	}

	virtual void UpdateAllScores(RegressionTree& tree)
	{
			for(ScoreTrackerPtr t : TrackedScores)
			{
				UpdateScores(t, tree);
			}
	}

	virtual void UpdateScores(ScoreTrackerPtr t, RegressionTree& tree)
	{
		if (t == TrainingScores)
		{
			if (AdjustTreeOutputsOverride != nullptr)
			{ //@TODO
			}
			else
			{
				t->AddScores(tree, TreeLearner->Partitioning, 1.0);
			}
		}
		else
		{
			t->AddScores(tree, 1.0);
		}
	}

public:
	IStepSearchPtr AdjustTreeOutputsOverride = nullptr;
	TreeLearnerPtr TreeLearner = nullptr; 
	ObjectiveFunctionPtr ObjectiveFunction = nullptr;
	Ensemble& Ensemble;
	double Smoothing = 0.0;
	vector<ScoreTrackerPtr> TrackedScores;
	ScoreTrackerPtr TrainingScores = nullptr;
	bool useFastTrainingScoresUpdate = true;
};

typedef shared_ptr<OptimizationAlgorithm> OptimizationAlgorithmPtr;
}  //----end of namespace gezi

#endif  //----end of OPTIMIZATION_ALGORITHM_H_
