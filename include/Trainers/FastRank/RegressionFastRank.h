/**
 *  ==============================================================================
 *
 *          \file   RegressionFastRank.h
 *
 *        \author   chenghuige
 *
 *          \date   2015-05-15 14:57:15.605539
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef REGRESSION_FAST_RANK_H_
#define REGRESSION_FAST_RANK_H_

#include "common_util.h"
#include "FastRank.h"
#include "RegressionFastRankArguments.h"
#include "RegressionObjectiveFunction.h"
namespace gezi {

	class RegressionFastRank : public FastRank
	{
	public:
		Fvec* TrainSetLabels;

		RegressionFastRank()
		{

		}

		virtual PredictionKind GetPredictionKind() override
		{
			return PredictionKind::Regression;
		}

		virtual PredictorPtr CreatePredictor() override
		{
			vector<OnlineRegressionTree> trees = _ensemble.ToOnline(TrainSet.Features);
			return make_shared<FastRankRegressionPredictor>(trees, TrainSet.FeatureNames());
		}

		virtual void ParseArgs() override
		{
			FastRank::ParseArgs();
		}

		virtual void Finalize(Instances& instances) override
		{
		
		}

		virtual void PrepareLabels() override
		{
			TrainSetLabels = &TrainSet.Ratings;
		}

		virtual FastRankArgumentsPtr GetArguments() override
		{
			return make_shared<RegressionFastRankArguments>();
		}

		virtual ObjectiveFunctionPtr ConstructObjFunc() override
		{
			return make_shared<RegressionObjectiveFunction>(TrainSet, *TrainSetLabels, *(((RegressionFastRankArguments*)(_args.get()))));
		}

		virtual void InitializeTests() override
		{

		}
	private:
	};


}  //----end of namespace gezi

#endif  //----end of REGRESSION_FAST_RANK_H_
