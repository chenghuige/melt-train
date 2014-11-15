/**
*  ==============================================================================
*
*          \file   LinearSVM.cpp
*
*        \author   chenghuige
*
*          \date   2014-04-08 19:36:17.509808
*
*  \Description:
*  ==============================================================================
*/

#ifndef LINEAR_S_V_M_CPP_
#define LINEAR_S_V_M_CPP_
#include "common_util.h"
#include "Trainers/SVM/LinearSVM.h"
#include "Trainers/SVM/BaseLineLinearSVM.h"

DECLARE_bool(calibrate);
DECLARE_string(calibrator);
DECLARE_uint64(rs);

DECLARE_bool(norm); //will speed up a if pre normalize and then --norm=0 for cross validation
DECLARE_string(normalizer);

DEFINE_int32(iter, 50000, "numIterExamples: Number of iterations");
DEFINE_double(lr, 0.001, "lambda: learning rate");
DEFINE_bool(project, false, "performProjection: Perform projection to unit-ball");
DEFINE_bool(nobias, false, "");
DEFINE_double(ss, 1.0, "sampleSize:");
DEFINE_double(sr, 0.001, "sampleRate:");
DEFINE_string(lt, "stochastic", "loopType: try roc or balanced");
DEFINE_string(trt, "peagsos", "trainerType: now only support peagsos");

namespace gezi {
	//@TODO 自动command 代码生成器
	void LinearSVM::ParseArgs() 
	{
		_args.calibrateOutput = FLAGS_calibrate;
		_args.calibratorName = FLAGS_calibrator;

		_args.normalizeFeatures = FLAGS_norm;

		_args.normalizerName = FLAGS_normalizer;

		_args.randSeed = FLAGS_rs;

		_args.numIterations = FLAGS_iter;
		_args.lambda = FLAGS_lr;
		_args.noBias = FLAGS_nobias;
		_args.sampleRate = FLAGS_sr;
		_args.sampleSize = FLAGS_ss;
		_args.performProjection = FLAGS_project;

		_args.loopType = FLAGS_lt;
		_args.trainerType = FLAGS_trt;
	}

	void BaseLineLinearSVM::ParseArgs()
	{
		_args.calibrateOutput = FLAGS_calibrate;
		_args.calibratorName = FLAGS_calibrator;

		_args.normalizeFeatures = FLAGS_norm;

		_args.normalizerName = FLAGS_normalizer;

		_args.randSeed = FLAGS_rs;

		_args.numIterations = FLAGS_iter;
		_args.lambda = FLAGS_lr;
	}


}  //----end of namespace gezi

#endif  //----end of LINEAR_S_V_M_CPP_
