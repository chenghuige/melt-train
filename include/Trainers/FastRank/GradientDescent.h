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
		Fvec _adjustedGradient; //AdjustTargetsAndSetWeights 如果改变了结果 那么返回这个值

		GradientDescent(gezi::Ensemble& ensemble, Dataset& trainData,
			Fvec& initTrainScores, IGradientAdjusterPtr gradientWrapper)
			: OptimizationAlgorithm(ensemble, trainData, initTrainScores), _gradientWrapper(gradientWrapper)
		{
		}

		virtual RegressionTree& TrainingIteration(BitArray& activeFeatures) override
		{
			RegressionTree tree = TreeLearner->FitTargets(activeFeatures, AdjustTargetsAndSetWeights());
			if (AdjustTreeOutputsOverride == nullptr)
			{ //如果父类ObjectiveFunction里面没有虚函数 不能使用dynamic_pointer_cast... @TODO
				(dynamic_pointer_cast<IStepSearch>(ObjectiveFunction))->AdjustTreeOutputs(tree, TreeLearner->Partitioning, *TrainingScores);
				/*((IStepSearch*)(ObjectiveFunction.get()))->AdjustTreeOutputs(tree, TreeLearner->Partitioning, *TrainingScores);*/ //@TODO 为什么用下面会运行失败？
			}
			else
			{//@TODO

			}
			if (Smoothing != 0.0)
			{
				SmoothTree(tree, Smoothing);
				_useFastTrainingScoresUpdate = false;
			}
			{
				UpdateAllScores(tree); //score traker的作用 感觉多叶子迭代 多树迭代后和tlc不一致可能和这里有关系
			}
			Ensemble.AddTree(tree);
			return Ensemble.Tree();
		}

		//@TODO
		virtual Fvec& AdjustTargetsAndSetWeights()
		{
			if (_gradientWrapper == nullptr)
			{
				return GetGradient();
			}
			Fvec* targetWeights = NULL;
			Fvec& targets = _gradientWrapper->AdjustTargetAndSetWeights(GetGradient(), *ObjectiveFunction, targetWeights);
			TreeLearner->TargetWeights = targetWeights;
			return targets;
		}

		virtual Fvec& GetGradient()
		{
			return ObjectiveFunction->GetGradient(TrainingScores->Scores);
		}

		virtual ScoreTrackerPtr ConstructScoreTracker(string name, Dataset& set, Fvec& initScores) override
		{
			return make_shared<ScoreTracker>(name, set, initScores);
		}


	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of GRADIENT_DESCENT_H_
