/**
 *  ==============================================================================
 *
 *          \file   RegressionObjectiveFunction.h
 *
 *        \author   chenghuige
 *
 *          \date   2015-05-15 14:56:03.858527
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef REGRESSION_OBJECTIVE_FUNCTION_H_
#define REGRESSION_OBJECTIVE_FUNCTION_H_

#include "Dataset.h"
#include "ObjectiveFunction.h"
#include "IStepSearch.h"
#include "BinaryClassificationGbdtArguments.h"
namespace gezi {

	//GetGradientInOneQuery 如果不想用虚函数或者std::function 也可以利用模板设计模式  泛型和多态是两种思路 可以考虑不同场景怎样更合适
	//泛型比较适合一个框架相同 核心函数交个不同的worker去给出不同实现 缺点是变量什么的 不同worker还需要都自己提供 或者函数接口多一些 不能继承了
	//template<typename GradientCalculator> 
	//class ObjectiveFunction //但是 这时候 GetGradientInOneQuery 需要接口增加gradient，和weight 或者GradientCalculator变量存储
	//{
	//      for...
	//          _gradientCalculator.GetGradientInOneQuery(query, scores, gradient, weight)
	//
	//  GradientCalculator _gradientCalculator;
	//}
	//class RegressionObjectiveFunction : public ObjectiveFunction, public IStepSearch
	class RegressionObjectiveFunction : public ObjectiveFunctionImpl<RegressionObjectiveFunction>, public IStepSearch
	{
	private:
		Fvec& Labels;

	public:
		RegressionObjectiveFunction(gezi::Dataset& trainSet, Fvec& trainSetLabels,
			RegressionGbdtArguments& args)
			: ObjectiveFunctionImpl(trainSet, args.learningRate, args.maxTreeOutput, args.derivativesSampleRate, args.bestStepRankingRegressionTrees, args.randSeed), Labels(trainSetLabels)
		{
			//GetGradientInOneQuery = [this](int query, const Fvec& scores)
			//{
			//	_gradient[query] = Labels[query] - scores[query];
			//};
		}

		virtual void AdjustTreeOutputs(RegressionTree& tree, DocumentPartitioning& partitioning, ScoreTracker& trainingScores) override
		{
			//@TODO AffineRegressionTree?
			for (int l = 0; l < tree.NumLeaves; l++)
			{
				Float output = 0.0;
				output = _learningRate * tree.GetOutput(l);

				if (output > _maxTreeOutput)
				{
					output = _maxTreeOutput;
				}
				else if (output < -_maxTreeOutput)
				{
					output = -_maxTreeOutput;
				}
				tree.SetOutput(l, output);
			}
		}


	public:
		//virtual void GetGradientInOneQuery(int query, const Fvec& scores) override
		void GetGradientInOneQuery(int query, const Fvec& scores)
		{
			_gradient[query] = Labels[query] - scores[query];
		}
	};

}  //----end of namespace gezi

#endif  //----end of REGRESSION_OBJECTIVE_FUNCTION_H_
