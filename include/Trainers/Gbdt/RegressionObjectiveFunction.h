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

	//GetGradientInOneQuery ����������麯������std::function Ҳ��������ģ�����ģʽ  ���ͺͶ�̬������˼· ���Կ��ǲ�ͬ��������������
	//���ͱȽ��ʺ�һ�������ͬ ���ĺ���������ͬ��workerȥ������ͬʵ�� ȱ���Ǳ���ʲô�� ��ͬworker����Ҫ���Լ��ṩ ���ߺ����ӿڶ�һЩ ���ܼ̳���
	//template<typename GradientCalculator> 
	//class ObjectiveFunction //���� ��ʱ�� GetGradientInOneQuery ��Ҫ�ӿ�����gradient����weight ����GradientCalculator�����洢
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
