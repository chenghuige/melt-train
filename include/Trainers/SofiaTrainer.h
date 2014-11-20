/**
 *  ==============================================================================
 *
 *          \file   Trainers/SofiaTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 05:15:35.156835
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__SOFIA_TRAINER_H_
#define TRAINERS__SOFIA_TRAINER_H_
#include "common_util.h"

#include "ThirdTrainer.h"
#include "Predictors/LinearPredictor.h"
#include "Predictors/SofiaPredictor.h" //not used, will gen LinearPredictor
#include "string2argcargv.h"

#define protected public
#define  private public
#include "sofia/sf-hash-weight-vector.h"
#include "sofia/sofia-ml-methods.h"
#include "sofia/sf-weight-vector.h"

namespace gezi {

	class SofiaTrainer : public ThirdTrainer
	{
	public:
		SofiaTrainer()
		{
			_classiferSettings = "--lambda 0.001 --iterations 50000";
		}

		virtual PredictorPtr CreatePredictor() override
		{
			return make_shared<LinearPredictor>(_weights, _bias, nullptr, nullptr, _featureNames, "Sofia");
		}

	protected:
		virtual void ShowHelp() override;

		SfDataSet Instances2SfDataSet(Instances& instances);
		virtual void Initialize(Instances& instances) override;
		virtual void InnerTrain(Instances& instances) override;
		virtual void Finalize(Instances& instances) override;

		/// <summary>
		/// Return the raw margin from the decision hyperplane
		/// </summary>		
		Float Margin(const Vector& features)
		{
			return _bias + _weights.dot(features);
		}

	private:

		FeatureNamesVector _featureNames;
		int _numFeatures;

		Vector _weights;
		Float _bias = 1.;

		SfWeightVector* w = NULL;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__SOFIA_TRAINER_H_
