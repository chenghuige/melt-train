/**
 *  ==============================================================================
 *
 *          \file   VWTrainer.cpp
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 16:23:09.804818
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef V_W_TRAINER_CPP_
#define V_W_TRAINER_CPP_

#include "vowpalwabbit/parser.h"
#include "vowpalwabbit/simple_label.h"
#include "vowpalwabbit/parse_args.h"
#include "vowpalwabbit/vw.h"
#include "vowpalwabbit/example.h"

#include "Predictors/VWPredictor.h"
#include "Trainers/VWTrainer.h"

namespace gezi {

	static VW::primitive_feature_space* VWTrainer::pFeatureSpace()
	{
		static thread_local VW::primitive_feature_space _pFeatureSpace;
		return &_pFeatureSpace;
	}

	PredictorPtr VWTrainer::CreatePredictor()
	{
		return make_shared<VWPredictor>(_vw, _pFeatureSpace, _normalizer, _calibrator, _featureNames);
	}

	example* VWTrainer::Instance2Example(InstancePtr instance, bool includeLabel)
	{
		int idx = 0;
		instance->features.ForEach([&](int index, Float value) {
			_pFeatureSpace->fs[idx].weight_index = index;
			_pFeatureSpace->fs[idx].x = value;
			idx++;
		});
		_pFeatureSpace->len = idx;
		example* ec = import_example(*_vw, _pFeatureSpace, 1);

		if (includeLabel)
		{
			Float label = instance->label <= 0 ? -1 : 1;
			VW::add_label(ec, label, instance->weight);
		}
		return ec;
	}

	void VWTrainer::Initialize(Instances& instances) 
	{
		string s = "";
		_vw = VW::initialize(s);
		_pFeatureSpace = pFeatureSpace();
		_pFeatureSpace->name = 'a';
		_pFeatureSpace->fs = new feature[instances.NumFeatures()];
	}


	void VWTrainer::InnerTrain(Instances& instances) 
	{
		for (InstancePtr instance : instances)
		{
			example* ec = Instance2Example(instance, true);
			_vw->learn(ec);
			//Pval(VW::get_prediction(ec));
			VW::finish_example(*_vw, ec);
		}
		//VW::finish(*_vw);
	}

	Float VWTrainer::Margin(InstancePtr instance)
	{
		example* ec = Instance2Example(instance, false);
		_vw->learn(ec); //@TODO TLC还是learn了 在predict的时候 check this
		Float output = VW::get_prediction(ec);
		VW::finish_example(*_vw, ec);
		return output;
	}

	void VWTrainer::Finalize(Instances& instances)
	{
		//FREE_ARRAY(_psf.fs);
		if (_calibrator != nullptr)
		{
			_calibrator->Train(instances, [this](InstancePtr instance) {
				return Margin(instance);
			});
		}
	}

}  //----end of namespace gezi

#endif  //----end of V_W_TRAINER_CPP_
