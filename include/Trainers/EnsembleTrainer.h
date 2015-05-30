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

	//��ǰ��ֻ���Ƕ����ͬ���͵ķ�����ѵ������Ҫ������������е���� ���ǰ�������ʱ��ֻ��һ��tranier��� ������ܳ�ͻ @TODO
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
