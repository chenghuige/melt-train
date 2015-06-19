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
		IGradientAdjusterPtr _gradientWrapper = nullptr; //@TODO
		Fvec _adjustedGradient; //AdjustTargetsAndSetWeights 如果改变了结果 那么返回这个值

		GradientDescent(gezi::Ensemble& ensemble, Dataset& trainData,
			Fvec& initTrainScores, IGradientAdjusterPtr gradientWrapper)
			: OptimizationAlgorithm(ensemble, trainData, initTrainScores), _gradientWrapper(gradientWrapper)
		{
		}

		virtual RegressionTree& TrainingIteration(const BitArray& activeFeatures) override
		{
			RegressionTree tree = TreeLearner->FitTargets(activeFeatures, AdjustTargetsAndSetWeights());
			//if (base.AdjustTreeOutputsOverride == null)
			//{
			//	if (!(base.ObjectiveFunction is IStepSearch))
			//	{
			//		throw new Exception("No AdjustTreeOutputs defined. Objective function should define IStepSearch or AdjustTreeOutputsOverride should be set");
			//	}
			//	(base.ObjectiveFunction as IStepSearch).AdjustTreeOutputs(tree, base.TreeLearner.Partitioning, base.TrainingScores);
			//}
			//else
			//{
			//	base.AdjustTreeOutputsOverride.AdjustTreeOutputs(tree, base.TreeLearner.Partitioning, base.TrainingScores);
			//}
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

			Fvec& targets = _gradientWrapper->AdjustTargetAndSetWeights(GetGradient(), *ObjectiveFunction, TreeLearner->TargetWeights);
			return targets;
		}

		virtual Fvec& GetGradient()
		{
			return ObjectiveFunction->GetGradient(TrainingScores->Scores);
		}

	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of GRADIENT_DESCENT_H_
