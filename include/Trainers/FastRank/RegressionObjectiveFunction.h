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
#include "BinaryClassificationFastRankArguments.h"
namespace gezi {

	class RegressionObjectiveFunction : public ObjectiveFunction, public IStepSearch
	{
	private:
		Fvec& Labels;

	public:
		RegressionObjectiveFunction(gezi::Dataset& trainSet, Fvec& trainSetLabels,
			RegressionFastRankArguments& args)
			: ObjectiveFunction(trainSet, args.learningRate, args.maxTreeOutput, args.derivativesSampleRate, args.bestStepRankingRegressionTrees, args.randSeed), Labels(trainSetLabels)
		{
					GetGradientInOneQuery = [this](int query, const Fvec& scores)
					{
					_gradient[query] = Labels[query] - scores[query];
					};
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


	protected:
		//virtual void GetGradientInOneQuery(int query, const Fvec& scores) override
		//{
		//	_gradient[query] = Labels[query] - scores[query];
		//}
	};

}  //----end of namespace gezi

#endif  //----end of REGRESSION_OBJECTIVE_FUNCTION_H_
