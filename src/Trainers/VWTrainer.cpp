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

	namespace {
		void dispatch_example(vw& all, example& ec) //copied from learner.cc
		{
			if (ec.test_only || !all.training)
				all.l->predict(ec);
			else
				all.l->learn(ec);
			all.l->finish_example(all, ec);
		}
	}
	static VW::primitive_feature_space* VWTrainer::pFeatureSpace()
	{
		static thread_local VW::primitive_feature_space _pFeatureSpace;
		return &_pFeatureSpace;
	}

	PredictorPtr VWTrainer::CreatePredictor()
	{
		_vw->training = false;
		Pval(_featureNames.size());
		return make_shared<VWPredictor>(_vw, _pFeatureSpace, _normalizer, _calibrator, _featureNames);
	}

	void VWTrainer::ShowHelp()
	{
		ParseArgs();
		if (!_vw)
		{
			_vw = VW::initialize(_classiferSettings);
		}
		cout << _vw->opts << endl;
	}

	example* VWTrainer::Instance2Example(InstancePtr instance, bool includeLabel)
	{
		int idx = 0;
		//注意如果是ForEach对于稠密数据是有问题的 需要ForEachNonZero
		instance->features.ForEachNonZero([&](int index, Float value) {
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
		_classiferSettings = _classiferSettings + " --random_seed " + STR(_randSeed);
		if (!_vw)
		{
			_vw = VW::initialize(_classiferSettings);
		}
		_vw->training = true;
		_pFeatureSpace = pFeatureSpace();
		_pFeatureSpace->name = 'a';
		_pFeatureSpace->fs = new feature[instances.NumFeatures()];
	}

	void VWTrainer::InnerTrain(Instances& instances)
	{
		for (InstancePtr instance : instances)
		{
			example* ec = Instance2Example(instance, true);
			dispatch_example(*_vw, *ec);
			//_vw->learn(ec);
			//Pval(VW::get_prediction(ec));
			//VW::finish_example(*_vw, ec);
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

	void VWTrainer::Finalize_(Instances& instances)
	{

	}
}  //----end of namespace gezi

#endif  //----end of V_W_TRAINER_CPP_
