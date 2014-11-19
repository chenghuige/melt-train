/**
 *  ==============================================================================
 *
 *          \file   Trainers/RandomTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-09-10 05:56:30.327518
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__RANDOM_TRAINER_H_
#define TRAINERS__RANDOM_TRAINER_H_

#include "MLCore/Trainer.h"
#include "Predictors/RandomPredictor.h"

namespace gezi {

class RandomTrainer : public Trainer
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

#endif  //----end of TRAINERS__RANDOM_TRAINER_H_
