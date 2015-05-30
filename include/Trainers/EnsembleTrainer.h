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

	//��ǰ��ֻ���Ƕ����ͬ���͵ķ�����ѵ������Ҫ������������е���� ���ǰ�������ʱ��ֻ��һ��tranier��� ������ܳ�ͻ @TODO
	class EnsembleTrainer : public Trainer
	{
	public:
		EnsembleTrainer() = default;
		virtual PredictorPtr CreatePredictor() override
		{
			return make_shared<EnsemblePredictor>(move(_predictors));
		}

		virtual void ParseArgs() override;

		//@TODO ������Ҫ��ķ����� ���粻�Ż����������ָ�����trainer �о�ûɶ��
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
		vector<string> _trainerNames = { "gbdt" };  //�м��ֲ�ͬ���͵�trainer
		vector<int> _numTrainers = { 5 }; //ÿ��trainer����Ŀ
		vector<TrainerPtr> _trainers;

		double _sampleRate = 0.7;
		size_t _maxCalibrationExamples = 1000000;

		vector<PredictorPtr> _predictors;
	private:
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__ENSEMBLE_TRAINER_H_
