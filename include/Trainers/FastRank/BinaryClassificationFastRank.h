/**
 *  ==============================================================================
 *
 *          \file   BinaryClassificationFastRank.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 14:32:08.893867
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef BINARY_CLASSIFICATION_FAST_RANK_H_
#define BINARY_CLASSIFICATION_FAST_RANK_H_
#include "common_util.h"
#include "FastRank.h"
#include "BinaryClassificationFastRankArguments.h"
#include "BinaryClassificationObjectiveFunction.h"
namespace gezi {

class BinaryClassificationFastRank : public FastRank
{
public:
	BitArray TrainSetLabels;

	BinaryClassificationFastRank()
	{
	
	}

	virtual PredictorPtr CreatePredictor() override
	{
		vector<OnlineRegressionTree> trees = _ensemble.GetOnlineRegressionTrees();
		return make_shared<FastRankPredictor>(trees, _calibrator, TrainSet.FeatureNames());
	}

	virtual void ParseArgs() override
	{
		FastRank::ParseArgs();
	}

	virtual void Finalize(Instances& instances) override
	{
		if (_args->calibrateOutput) //@TODO to trainer
		{
			_calibrator = CalibratorFactory::CreateCalibrator(_args->calibratorName);
			PVAL((_calibrator == nullptr));
			if (_calibrator != nullptr)
			{
				_calibrator->Train(GetTrainSetScores(), TrainSetLabels, TrainSet.SampleWeights);
			}
		}
	}

	virtual void PrepareLabels() override
	{
		/*TrainSetLabels = from(TrainSet.Ratings)
			>> select([this](short a) { return (bool)(a >= (dynamic_pointer_cast<BinaryClassificationFastRankArguments>(_args))->smallestPositive	); })
			>> to_vector();*/

		TrainSetLabels = from(TrainSet.Ratings)
			>> select([this](Float a) { return (bool)(a >= ((BinaryClassificationFastRankArguments*)(_args.get()))->smallestPositive); })
			>> to_vector();
	}

	virtual FastRankArgumentsPtr GetArguments() override
	{
		return make_shared<BinaryClassificationFastRankArguments>();
	}

	virtual ObjectiveFunctionPtr ConstructObjFunc() override
	{
		/*return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *(dynamic_pointer_cast<BinaryClassificationFastRankArguments>(_args)));*/

		return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *(((BinaryClassificationFastRankArguments*)(_args.get()))));
	}

	//virtual void InitializeTests() override
	//{

	//}
private:
};

}  //----end of namespace gezi

#endif  //----end of BINARY_CLASSIFICATION_FAST_RANK_H_
