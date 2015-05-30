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

	//当前先只考虑多个相同类型的分类器训练，主要问题解析命令行的设计 都是按照运行时候只有一个tranier设计 多个可能冲突 @TODO
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
		vector<string> _trainerNames;  //有几种trainer
		vector<int> _numTrainers; //每种trainer的数目
		vector<TrainerPtr> _trainers;
	private:
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__ENSEMBLE_TRAINER_H_
