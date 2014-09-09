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

#include "ProgressBar.h"
#include "MLCore/Trainer.h"
#include "Prediction/Instances/Instances.h"
#include "Numeric/Vector/Vector.h"
#include "Predictors/RandomPredictor.h"
#include "Prediction/Normalization/NormalizerFactory.h"
#include "Prediction/Calibrate/CalibratorFactory.h"

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
