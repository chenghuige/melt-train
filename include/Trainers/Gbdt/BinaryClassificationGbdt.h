/**
 *  ==============================================================================
 *
 *          \file   BinaryClassificationGbdt.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 14:32:08.893867
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef BINARY_CLASSIFICATION_GBDT_H_
#define BINARY_CLASSIFICATION_GBDT_H_
#include "common_util.h"
#include "Gbdt.h"
#include "BinaryClassificationGbdtArguments.h"
#include "BinaryClassificationObjectiveFunction.h"
#include "rabit_util.h"
DECLARE_int32(distributeMode);
namespace gezi {

	class BinaryClassificationGbdt : public Gbdt
	{
	public:
		BitArray TrainSetLabels;

		BinaryClassificationGbdt()
		{
		}

		virtual PredictorPtr CreatePredictor(vector<OnlineRegressionTree>& trees) override
		{
			auto predictor = make_shared<GbdtPredictor>(trees);
			predictor->SetCalibrator(_calibrator);
			return predictor;
		}

		virtual void ParseArgs() override
		{
			Gbdt::ParseArgs();
			_args = static_cast<BinaryClassificationGbdtArguments*>(Gbdt::_args.get());
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
					if (_args->numBags == 1)
					{
						if (Rabit::GetWorldSize() > 1 && FLAGS_distributeMode == 2)
						{
							Rabit::Allgather(TrainScores);
							Rabit::Allgather(TrainSet.SampleWeights);
							Rabit::Allgather(TrainSetLabels);
						}
						_calibrator->Train(TrainScores, TrainSetLabels, TrainSet.SampleWeights);
					}
					else if (_selfEvaluate)
					{
						for (auto& item : Scores.back())
						{
							item /= (double)_args->numBags;
						}
						_calibrator->Train(Scores.back(), *InputInstances);
					}
					else
					{
						_calibrator->Train(*InputInstances,
							[&](InstancePtr instance) { return _ensemble.GetOutput(instance); },
							_args->maxCalibrationExamples);
						/*((BinaryClassificationGbdtArguments*)(_args.get()))->maxCalibrationExamples);*/
						/*			include / c++ / 4.8.2 / bits / shared_ptr.h:449 : 50 : error : cannot dynamic_cast '(& __r)->std::shared_ptr<gezi::GbdtArguments>::<anonymous>.std::__shared_ptr<_Tp, _Lp>::get<gezi::GbdtArguments, (__gnu_cxx::_Lock_policy)2u>()' (of type 'struct gezi::GbdtArguments*') to type 'struct gezi::BinaryClassificationGbdtArguments*' (source type is not polymorphic)
									if (_Tp* __p = dynamic_cast<_Tp*>(__r.get()))*/
						//dynamic_pointer_cast<BinaryClassificationGbdtArguments>(_args)->maxCalibrationExamples);
					}
				}
			}
		}

		virtual void PrepareLabels() override
		{
			/*TrainSetLabels = from(TrainSet.Ratings)
				>> select([this](short a) { return (bool)(a >= (dynamic_pointer_cast<BinaryClassificationGbdtArguments>(_args))->smallestPositive	); })
				>> to_vector();*/

			TrainSetLabels = from(TrainSet.Ratings)
				/*>> select([this](Float a) { return (bool)(a >= ((BinaryClassificationGbdtArguments*)(_args.get()))->smallestPositive); })*/
				>> select([this](Float a) { return (bool)(a >= _args->smallestPositive); })
				>> to_vector();
		}

		virtual GbdtArgumentsPtr CreateArguments() override
		{
			return make_shared<BinaryClassificationGbdtArguments>();
		}

		virtual ObjectiveFunctionPtr ConstructObjFunc() override
		{
			/*return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *(dynamic_pointer_cast<BinaryClassificationGbdtArguments>(_args)));*/

			/*		return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *(((BinaryClassificationGbdtArguments*)(_args.get()))));*/
			return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *_args);
		}

		//virtual void InitializeTests() override
		//{

		//}
	private:
		BinaryClassificationGbdtArguments* _args = NULL; //覆盖掉Fastrank的
	};

}  //----end of namespace gezi

#endif  //----end of BINARY_CLASSIFICATION_GBDT_H_
