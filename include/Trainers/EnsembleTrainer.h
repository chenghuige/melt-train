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
namespace gezi {

	//��ǰ��ֻ���Ƕ����ͬ���͵ķ�����ѵ������Ҫ������������е���� ���ǰ�������ʱ��ֻ��һ��tranier��� ������ܳ�ͻ @TODO
	class EnsembleTrainer : public Trainer
	{
	public:
		virtual PredictorPtr CreatePredictor() override
		{
			vector<PredictorPtr> predictors;
			for (auto& trainer : _trainers)
			{
				predictors.push_back(trainer->CreatePredictor());
			}
			return make_shared<EnsemblePredictor>(move(predictors));
		}

	protected:
		virtual void Init()
		{
			for (size_t i = 0; i < _trainerNames.size(); i++)
			{
				for (int j = 0; j < _numTrainers[i]; j++)
				{
					_trainers.push_back(TrainerFactory::CreateTrainer(_trainerNames[i]));
				}
			}
		}
	protected:
		vector<string> _trainerNames;  //�м���trainer
		vector<int> _numTrainers; //ÿ��trainer����Ŀ
		vector<TrainerPtr> _trainers;
	private:
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__ENSEMBLE_TRAINER_H_
