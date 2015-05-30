/**
 *  ==============================================================================
 *
 *          \file   Trainers/EnsembleTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2015-04-16 20:03:19.082694
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__ENSEMBLE_TRAINER_H_
#define TRAINERS__ENSEMBLE_TRAINER_H_

#include "MLCore/Trainer.h"
#include "MLCore/TrainerFactory.h"
#include "Predictors/EnsemblePredictor.h"
#include "Prediction/Instances/instances_util.h"
namespace gezi {

	//当前先只考虑多个相同类型的分类器训练，主要问题解析命令行的设计 都是按照运行时候只有一个tranier设计 多个可能冲突 @TODO
	class EnsembleTrainer : public Trainer
	{
	public:
		EnsembleTrainer() = default;
		virtual PredictorPtr CreatePredictor() override
		{
			return make_shared<EnsemblePredictor>(move(_predictors));
		}

		virtual void ParseArgs() override;

		//@TODO 采样需要别的方法吗 比如不放回随机的随机分给几个trainer 感觉没啥用
		virtual void Train(Instances& instances)  override
		{
			Init();
			for (size_t i = 0; i < _trainers.size(); i++)
			{
				Instances partionInstances = InstancesUtil::GenPartionInstances(instances, *_rand, _sampleRate);
				_trainers[i]->Train(partionInstances);
			}
			Finalize(instances);
		}

		virtual void Finalize(Instances& instances) override
		{
			for (auto& trainer : _trainers)
			{
				_predictors.push_back(trainer->CreatePredictor());
			}

			if (_calibrator != nullptr)
			{
				VLOG(0) << "Calirate once for the final ensemble model";
				_calibrator->Train(instances, [this](InstancePtr instance) {
					return Margin(instance); },
						_maxCalibrationExamples);
			}
		}

	protected:
		virtual void Init()
		{
			ParseArgs();
			for (size_t i = 0; i < _trainerNames.size(); i++)
			{
				for (int j = 0; j < _numTrainers[i]; j++)
				{
					_trainers.push_back(TrainerFactory::CreateTrainer(_trainerNames[i]));
				}
			}
		}

		virtual Float Margin(InstancePtr instance)
		{
			double output = 0;
#pragma omp parallel for reduction(+: output)
			for (size_t i = 0; i < _predictors.size(); i++)
			{
				output += _predictors[i]->Output(instance);
			}
			return output / _predictors.size();
		}
	protected:
		vector<string> _trainerNames = { "gbdt" };  //有几种不同类型的trainer
		vector<int> _numTrainers = { 5 }; //每种trainer的数目
		vector<TrainerPtr> _trainers;

		double _sampleRate = 0.7;
		size_t _maxCalibrationExamples = 1000000;

		vector<PredictorPtr> _predictors;
	private:
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__ENSEMBLE_TRAINER_H_
