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
#include "Predictors/RandomPredictor.h"
namespace gezi {

	//当前先只考虑多个相同类型的分类器训练，主要问题解析命令行的设计 都是按照运行时候只有一个tranier设计 多个可能冲突 @TODO
	class EnsembleTrainer : public Trainer
	{
	public:
		virtual PredictorPtr CreatePredictor() override
		{
			return make_shared<RandomPredictor>();
		}
	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__ENSEMBLE_TRAINER_H_
