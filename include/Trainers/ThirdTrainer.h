/**
 *  ==============================================================================
 *
 *          \file   Trainers/ThirdTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 16:33:18.028897
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__THIRD_TRAINER_H_
#define TRAINERS__THIRD_TRAINER_H_

#include "MLCore/Trainer.h"
#include "Predictors/LinearPredictor.h"
#include "string2argcargv.h"

namespace gezi {

	class ThirdTrainer : public Trainer
	{
	public:
		virtual void ParseArgs() override;
		virtual void Init() override;

		//Ä¬ÈÏ·µ»ØLinearPredictor
		virtual PredictorPtr CreatePredictor() override
		{
			return make_shared<LinearPredictor>(_weights, _bias, _normalizer, _calibrator, _featureNames, GetPredictorName());
		}

		virtual string GetPredictorName()
		{
			return "";
		}
	protected:
		//----------for LinearPredictor
		Vector _weights;
		Float _bias = 1.;

		string _classiferSettings;
		unsigned _randSeed = 0;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__THIRD_TRAINER_H_
