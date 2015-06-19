/**
 *  ==============================================================================
 *
 *          \file   RegressionGbdt.h
 *
 *        \author   chenghuige
 *
 *          \date   2015-05-15 14:57:15.605539
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef REGRESSION_GBDT_H_
#define REGRESSION_GBDT_H_

#include "common_util.h"
#include "Gbdt.h"
#include "RegressionGbdtArguments.h"
#include "RegressionObjectiveFunction.h"
namespace gezi {

	class RegressionGbdt : public Gbdt
	{
	public:
		Fvec* TrainSetLabels;

		RegressionGbdt()
		{

		}

		virtual PredictionKind GetPredictionKind() override
		{
			return PredictionKind::Regression;
		}

		virtual LossKind GetLossKind() override
		{
			return LossKind::Squared;
		}

		virtual PredictorPtr CreatePredictor(vector<OnlineRegressionTree>& trees) override
		{
			auto predictor = make_shared<GbdtRegressionPredictor>(trees);
			return predictor;
		}

		virtual void ParseArgs() override
		{
			Gbdt::ParseArgs();
			_args = static_cast<RegressionGbdtArguments*>(Gbdt::_args.get());
			ParseRegressionArgs();
		}

		void ParseRegressionArgs()
		{

		}

		virtual void Finalize(Instances& instances) override
		{
		
		}

		virtual void PrepareLabels() override
		{
			TrainSetLabels = &TrainSet.Ratings;
		}

		virtual GbdtArgumentsPtr CreateArguments() override
		{
			return make_shared<RegressionGbdtArguments>();
		}

		virtual ObjectiveFunctionPtr ConstructObjFunc() override
		{
			/*return make_shared<RegressionObjectiveFunction>(TrainSet, *TrainSetLabels, *(((RegressionGbdtArguments*)(_args.get()))));*/
			return make_shared<RegressionObjectiveFunction>(TrainSet, *TrainSetLabels, *_args);
		}

		//virtual void InitializeTests() override
		//{

		//}
	private:
		RegressionGbdtArguments* _args = NULL;
	};


}  //----end of namespace gezi

#endif  //----end of REGRESSION_GBDT_H_
