/**
 *  ==============================================================================
 *
 *          \file   Trainers/VWTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-19 15:40:10.886164
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__V_W_TRAINER_H_
#define TRAINERS__V_W_TRAINER_H_

#include "MLCore/Trainer.h"
#include "Predictors/VWPredictor.h"

#include "vowpalwabbit/parser.h"
#include "vowpalwabbit/simple_label.h"
#include "vowpalwabbit/parse_args.h"
#include "vowpalwabbit/vw.h"
#include "vowpalwabbit/example.h"

namespace gezi {

//暂时不支持VW解析输入 @TODO
class VWTrainer : public Trainer
{
public:
	virtual PredictorPtr CreatePredictor() override
	{
		return make_shared<VWPredictor>(_vw, _pFeatureSpace);
	}
protected:
	example* Instance2Example(InstancePtr instance, bool includeLabel)
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
protected:
	virtual void Initialize(Instances& instances) override
	{
		string s = "";
		_vw = VW::initialize(s);
		_pFeatureSpace = &pFeatureSpace();
		_pFeatureSpace->name = 'a';
		_pFeatureSpace->fs = new feature[instances.NumFeatures()];
	}

	virtual void InnerTrain(Instances& instances) override
	{
		for(InstancePtr instance : instances)
		{
				example* ec = Instance2Example(instance, true);
				_vw->learn(ec);
				//Pval(VW::get_prediction(ec));
				VW::finish_example(*_vw, ec);
		}
		//VW::finish(*_vw);
	}

	virtual void Finalize(Instances& instances) override
	{
		//FREE_ARRAY(_psf.fs);
	}


public:
	static VW::primitive_feature_space& pFeatureSpace()
	{
		static thread_local VW::primitive_feature_space _pFeatureSpace;
		return _pFeatureSpace;
	}
	VW::primitive_feature_space* _pFeatureSpace;
	vw* _vw = NULL;
	string modelFile;
	string readableModelFile;
};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__V_W_TRAINER_H_
