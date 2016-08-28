/**
 *  ==============================================================================
 *
 *          \file   RankingGbdt.h
 *
 *        \author   chenghuige
 *
 *          \date   2016-06-19 17:19:59.671134
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef GEZI_RANKING_GBDT_H_
#define GEZI_RANKING_GBDT_H_

#include "common_util.h"
#include "Gbdt.h"
#include "RankingGbdtArguments.h"
#include "LambdaRankObjectiveFunction.h"
namespace gezi {

	class RankingGbdt : public Gbdt
	{
	public:
		Fvec* TrainSetLabels;

		RankingGbdt()
		{

		}

		virtual PredictionKind GetPredictionKind() override
		{
			return PredictionKind::Ranking;
		}

		virtual LossKind GetLossKind() override
		{
			return LossKind::Squared; //@TODO
		}

		virtual PredictorPtr CreatePredictor(vector<OnlineRegressionTree>& trees) override
		{
			auto predictor = make_shared<GbdtRankingPredictor>(trees);
			return predictor;
		}

		virtual void ParseArgs() override
		{
			Gbdt::ParseArgs();
			_args = static_cast<RankingGbdtArguments*>(Gbdt::_args.get());
			ParseRankingArgs();
		}

		void ParseRankingArgs();

		virtual void Finalize(Instances& instances) override
		{

		}

		virtual void PrepareLabels() override
		{
			TrainSetLabels = &TrainSet.Ratings;
		}

		virtual GbdtArgumentsPtr CreateArguments() override
		{
			return make_shared<RankingGbdtArguments>();
		}

		virtual ObjectiveFunctionPtr ConstructObjFunc() override
		{
			/*return make_shared<RegressionObjectiveFunction>(TrainSet, *TrainSetLabels, *(((RegressionGbdtArguments*)(_args.get()))));*/
			return make_shared<LambdaRankObjectiveFunction>(TrainSet, *TrainSetLabels, *_args);
		}

		//virtual void InitializeTests() override
		//{

		//}
	private:
		RankingGbdtArguments* _args = NULL;
	};


}  //----end of namespace gezi


#endif  //----end of GEZI_RANKING_GBDT_H_
