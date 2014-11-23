/**
 *  ==============================================================================
 *
 *          \file   ThirdTrainer.cpp
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 06:01:08.123837
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef THIRD_TRAINER_CPP_
#define THIRD_TRAINER_CPP_

#include "Trainers/ThirdTrainer.h"
DECLARE_bool(norm); //will speed up a if pre normalize and then --norm=0 for cross validation
DECLARE_string(normalizer);

DECLARE_bool(calibrate);
DECLARE_string(calibrator);
DECLARE_uint64(numCali);

DECLARE_string(cls);
DECLARE_uint64(rs);

namespace gezi {

	//所有第三方的共同特性抽出
	void ThirdTrainer::ParseArgs()
	{
		PVAL(FLAGS_cls);
		if (!FLAGS_cls.empty())
		{
			_classiferSettings = gezi::replace_chars(FLAGS_cls, "=,", ' ');
		}
		Pval(_classiferSettings);
		_randSeed = FLAGS_rs;
		_maxCalibrationExamples = FLAGS_numCali;
	}

	void ThirdTrainer::Init()
	{
		ParseArgs();

		PVAL(_randSeed);
		if (FLAGS_norm) //@TODO to trainer
		{
			_normalizer = NormalizerFactory::CreateNormalizer(FLAGS_normalizer);
		}
		PVAL((_normalizer == nullptr));

		if (FLAGS_calibrate) //@TODO to trainer
		{
			_calibrator = CalibratorFactory::CreateCalibrator(FLAGS_calibrator);
		}
		PVAL((_calibrator == nullptr));
	}

}  //----end of namespace gezi

#endif  //----end of THIRD_TRAINER_CPP_
