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

		virtual PredictorPtr CreatePredictor(vector<OnlineRegressionTree>& trees) override
		{
			auto predictor = make_shared<FastRankPredictor>(trees);
			predictor->SetCalibrator(_calibrator);
			return predictor;
		}

		virtual void ParseArgs() override
		{
			FastRank::ParseArgs();
			_args = static_cast<BinaryClassificationFastRankArguments*>(FastRank::_args.get());
			ParseClassificationArgs();
		}
		void ParseClassificationArgs();

		//@TODO 类似Binary这种也可以选配其它的LossFunction?
		virtual LossKind GetLossKind() override
		{
			return LossKind::Logistic;
		}

		virtual void Finalize(Instances& instances) override
		{
			if (_args->calibrateOutput) //@TODO to trainer
			{
				_calibrator = CalibratorFactory::CreateCalibrator(_args->calibratorName);
				PVAL((_calibrator == nullptr));
				if (_calibrator != nullptr)
				{
					/*	_calibrator->Train(GetTrainSetScores(), TrainSetLabels, TrainSet.SampleWeights);*/
					//如果不是bagging 有sampling fraq 完全正确 并且最快，如果有sampling那么最后记录的数据 不是最终输出的output对应的。。
					if (_args->numBags == 1 || _args->baggingTrainFraction >= 1.0)
					{
						_calibrator->Train(TrainScores, TrainSetLabels, TrainSet.SampleWeights);
					}
					else
					{
						_calibrator->Train(*InputInstances,
							[&](InstancePtr instance) { return _ensemble.GetOutput(instance); },
							_args->maxCalibrationExamples);
							/*((BinaryClassificationFastRankArguments*)(_args.get()))->maxCalibrationExamples);*/
							/*			include / c++ / 4.8.2 / bits / shared_ptr.h:449 : 50 : error : cannot dynamic_cast '(& __r)->std::shared_ptr<gezi::FastRankArguments>::<anonymous>.std::__shared_ptr<_Tp, _Lp>::get<gezi::FastRankArguments, (__gnu_cxx::_Lock_policy)2u>()' (of type 'struct gezi::FastRankArguments*') to type 'struct gezi::BinaryClassificationFastRankArguments*' (source type is not polymorphic)
										if (_Tp* __p = dynamic_cast<_Tp*>(__r.get()))*/
										//dynamic_pointer_cast<BinaryClassificationFastRankArguments>(_args)->maxCalibrationExamples);
						//@TODO 看上去 还是有自己的_args比较好 不要用继承 分开
					}
				}
			}
		}

		virtual void PrepareLabels() override
		{
			/*TrainSetLabels = from(TrainSet.Ratings)
				>> select([this](short a) { return (bool)(a >= (dynamic_pointer_cast<BinaryClassificationFastRankArguments>(_args))->smallestPositive	); })
				>> to_vector();*/

			TrainSetLabels = from(TrainSet.Ratings)
				/*>> select([this](Float a) { return (bool)(a >= ((BinaryClassificationFastRankArguments*)(_args.get()))->smallestPositive); })*/
				>> select([this](Float a) { return (bool)(a >= _args->smallestPositive); })
				>> to_vector();
		}

		virtual FastRankArgumentsPtr CreateArguments() override
		{
			return make_shared<BinaryClassificationFastRankArguments>();
		}

		virtual ObjectiveFunctionPtr ConstructObjFunc() override
		{
			/*return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *(dynamic_pointer_cast<BinaryClassificationFastRankArguments>(_args)));*/

	/*		return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *(((BinaryClassificationFastRankArguments*)(_args.get()))));*/
			return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *_args);
		}

		//virtual void InitializeTests() override
		//{

		//}
	private:
		BinaryClassificationFastRankArguments* _args = NULL; //覆盖掉Fastrank的
	};

}  //----end of namespace gezi

#endif  //----end of BINARY_CLASSIFICATION_FAST_RANK_H_
