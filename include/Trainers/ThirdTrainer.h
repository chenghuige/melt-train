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

		
		virtual string GetPredictorName()
		{
			return "";
		}

		virtual void Finalize(Instances& instances) override
		{
			Finalize_(instances);
			if (_calibrator != nullptr)
			{
				_calibrator->Train(instances, [this](InstancePtr instance) {
					return Margin(instance); },
						_maxCalibrationExamples);
			}
		}

	protected:
		virtual void Finalize_(Instances& instances)
		{

		}
		
		virtual Float Margin(InstancePtr instance)
		{
			return 0;
		}
	protected:
		string _classiferSettings;
		unsigned _randSeed = 0; //@TODO all unsigned rand seed to size_t ?

		size_t _maxCalibrationExamples = 1000000;
	};

	class LinearThirdTrainer : public ThirdTrainer
	{
	public:
		//Ä¬ÈÏ·µ»ØLinearPredictor
		virtual PredictorPtr CreatePredictor() override
		{
			return make_shared<LinearPredictor>(_weights, _bias, _normalizer, _calibrator, _featureNames, GetPredictorName());
		}

		virtual void Finalize(Instances& instances) override
		{
			Finalize_(instances);
			if (_calibrator != nullptr)
			{
				_calibrator->Train(instances, [this](InstancePtr instance) {
					return Margin(instance->features); },
						_maxCalibrationExamples);
			}
		}
	protected:
		/// <summary>
		/// Return the raw margin from the decision hyperplane
		/// </summary>		
		Float Margin(const Vector& features)
		{
			return _bias + _weights.dot(features);
		}

	protected:
		//----------for LinearPredictor
		Vector _weights;
		Float _bias = 0.;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__THIRD_TRAINER_H_
