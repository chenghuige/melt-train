/**
 *  ==============================================================================
 *
 *          \file   LambdaRankObjectiveFunction.h
 *
 *        \author   chenghuige
 *
 *          \date   2016-06-19 17:42:08.102975
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef GEZI_LAMBDA_RANK_OBJECTIVE_FUNCTION_H_
#define GEZI_LAMBDA_RANK_OBJECTIVE_FUNCTION_H_

#include "Dataset.h"
#include "ObjectiveFunction.h"
#include "RankingObjectiveFunction.h"
#include "IStepSearch.h"
#include "RankingGbdtArguments.h"

namespace gezi {
	//@TODO now only copy gbdt regression
	class LambdaRankObjectiveFunction :public ObjectiveFunctionImpl<LambdaRankObjectiveFunction>, 
		public RankingObjectiveFunction,
		public IStepSearch
{
public:
	LambdaRankObjectiveFunction(gezi::Dataset& trainSet, Fvec& trainSetLabels,
		RankingGbdtArguments& args)
		: ObjectiveFunctionImpl(trainSet, args.learningRate, args.maxTreeOutput, args.derivativesSampleRate, args.bestStepRankingRegressionTrees, args.randSeed), RankingObjectiveFunction(trainSetLabels)
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

protected:
private:

};

}  //----end of namespace gezi

#endif  //----end of GEZI_LAMBDA_RANK_OBJECTIVE_FUNCTION_H_
