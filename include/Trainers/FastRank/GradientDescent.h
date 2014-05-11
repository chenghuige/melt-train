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
#include "IGradientAdjuster.h"
#include "Ensemble.h"
#include "Dataset.h"
#include "ObjectiveFunction.h"
namespace gezi {

	class GradientDescent : public OptimizationAlgorithm
	{
	public:
		IGradientAdjusterPtr _gradientWrapper; //@TODO
		dvec _adjustedGradient; //AdjustTargetsAndSetWeights 如果改变了结果 那么返回这个值

		GradientDescent(::Ensemble& ensemble, Dataset& trainData, 
			dvec& initTrainScores, IGradientAdjusterPtr gradientWrapper)
			: OptimizationAlgorithm(ensemble, trainData, initTrainScores), _gradientWrapper(gradientWrapper)
		{
		}

		virtual RegressionTree& TrainingIteration() override
		{
			RegressionTree tree = TreeLearner->FitTargets(AdjustTargetsAndSetWeights());
			if (AdjustTreeOutputsOverride == nullptr)
			{
				((IStepSearch*)(ObjectiveFunction.get()))->AdjustTreeOutputs(tree, TreeLearner->Partitioning, *TrainingScores);
			}
			else
			{//@TODO

			}
			if (Smoothing != 0.0)
			{
				SmoothTree(tree, Smoothing);
				useFastTrainingScoresUpdate = false;
			}
			UpdateAllScores(tree);
			Ensemble.AddTree(tree);
			return Ensemble.Tree();
		}

		//@TODO
		virtual dvec& AdjustTargetsAndSetWeights()
		{
			return GetGradient();
		}

		virtual dvec& GetGradient()
		{
			return ObjectiveFunction->GetGradient(TrainingScores->Scores);
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
