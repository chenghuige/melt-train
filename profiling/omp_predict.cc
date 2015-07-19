/**
 *  ==============================================================================
 *
 *          \file   omp_predict.cc
 *
 *        \author   chenghuige
 *
 *          \date   2015-06-04 12:45:36.871187
 *
 *  \Description:
 *
 *  ==============================================================================
 */

#define _DEBUG
#define private public
#define protected public
#include "common_util.h"
#include "Prediction/Instances/instances_util.h"
#include "MLCore/PredictorFactory.h"
#include "Utils/PredictorUtils.h"

using namespace std;
using namespace gezi;
DECLARE_string(i);
DECLARE_string(m);

Instances _instances;
PredictorPtr _predictor;
TesterPtr _tester;

TEST(tester, func)
{
	_tester->Test(_instances, _predictor, "./result.txt");
}

TEST(predict, func)
{
	for (size_t i = 0; i < _instances.size(); i++)
	{
		double output;
		double probability = _predictor->Predict(_instances[i], output);
	}
}

TEST(predict2, func)
{
	dvec outputs(_instances.size()), probabilities(_instances.size());
	for (size_t i = 0; i < _instances.size(); i++)
	{
		probabilities[i] = _predictor->Predict(_instances[i], outputs[i]);
	}
}

TEST(omp_predict, func)
{
	dvec outputs(_instances.size()), probabilities(_instances.size());
#pragma omp parallel for
	for (size_t i = 0; i < _instances.size(); i++)
	{
		probabilities[i] = _predictor->Predict(_instances[i], outputs[i]);
	}
}

TEST(omp_predict_static, func)
{
	dvec outputs(_instances.size()), probabilities(_instances.size());
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < _instances.size(); i++)
	{
		probabilities[i] = _predictor->Predict(_instances[i], outputs[i]);
	}
}

TEST(omp_predict_dynamic, func)
{
	dvec outputs(_instances.size()), probabilities(_instances.size());
#pragma omp parallel for schedule(dynamic)
	for (size_t i = 0; i < _instances.size(); i++)
	{
		probabilities[i] = _predictor->Predict(_instances[i], outputs[i]);
	}
}

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
	int s = google::ParseCommandLineFlags(&argc, &argv, false);

	_instances = create_instances(FLAGS_i);
	_predictor = PredictorFactory::LoadPredictor(FLAGS_m);

	_tester = PredictorUtils::GetTester(_predictor);

	return RUN_ALL_TESTS();
}
