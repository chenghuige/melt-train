/**
 *  ==============================================================================
 *
 *          \file   EnsembleTrainer.cpp
 *
 *        \author   chenghuige
 *
 *          \date   2015-05-30 22:21:52.085835
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef ENSEMBLE_TRAINER_CPP_
#define ENSEMBLE_TRAINER_CPP_
#include "common_util.h"
#include "Trainers/EnsembleTrainer.h"

DEFINE_string(trainers, "gbdt", "trainerNames|like gbdt,linearsvm");
DEFINE_string(numTrainers, "5", "numTrainers|like 2,1, with -trainer gbdt,linearsvm means train 2 gbdt and 1 linearsvm");
DEFINE_double(srate, 0.7, "sampleRate|replace sampling rate");
DEFINE_bool(edistribute, false, "AllowEnsembleDistribute: now only for ensemble trainer when distibute is set to false will not train different trainer parallel, if parallel train and the sub trainer parallel train it's self may cause error");

DECLARE_bool(bstrap);

DECLARE_uint64(rs);
DECLARE_uint64(numCali);
DECLARE_bool(calibrate);
DECLARE_string(calibrator);
namespace gezi {

	void EnsembleTrainer::ParseArgs()
	{
		_trainerNames = gezi::split(FLAGS_trainers, ',');
		gezi::convert(gezi::split(FLAGS_numTrainers, ','), _numTrainers);
		gezi::print(_trainerNames, _numTrainers);
		_sampleRate = FLAGS_srate;

		_rand = make_shared<Random>(random_engine(FLAGS_rs));
		_randSeed = FLAGS_rs;
		_maxCalibrationExamples = FLAGS_numCali;

		_bootStrap = FLAGS_bstrap;

		_allowDistribute = FLAGS_edistribute;

		if (_trainerNames.size() == 1)
		{
			VLOG(0) << "Only one kind trainer disable calibrate during each trainer training process";
			if (FLAGS_calibrate) //@TODO to trainer
			{
				_calibrator = CalibratorFactory::CreateCalibrator(FLAGS_calibrator); 
				FLAGS_calibrate = false;
			}
		}
	}
}  //----end of namespace gezi

#endif  //----end of ENSEMBLE_TRAINER_CPP_
