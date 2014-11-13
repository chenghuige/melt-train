/** 
 *  ==============================================================================
 * 
 *          \file   test_train_test.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2014-11-10 15:27:01.929067
 *  
 *  \Description:
 *
 *  ==============================================================================
 */

#define _DEBUG
#define private public
#define protected public
#include "common_util.h"
//#include "Trainers/FastRank/BinaryClassificationFastRank.h"
#include "MLCore/TrainerFactory.h"
#include "Prediction/Instances/instances_util.h"
#include "Utils/PredictorUtils.h"
#include "Run/Melt.h"
using namespace std;
using namespace gezi;
DEFINE_int32(vl, 0, "vlog level");
DEFINE_string(in, "./data/feature.normed.train", "train file");
DEFINE_string(out, "./data/feature.normed.test", "test file");
DEFINE_string(result, "./result.txt", "result file");
DECLARE_string(cl);
DECLARE_bool(calibrate);
DECLARE_bool(norm);

TEST(train_test, func)
{
	FLAGS_norm = false;
	FLAGS_calibrate = false;
	auto trainer = TrainerFactory::CreateTrainer(FLAGS_cl);
	CHECK_NE((trainer == nullptr), true);
	auto instances = create_instances(FLAGS_in);
	trainer->Train(instances);

	auto predictor = trainer->CreatePredictor();

	auto testInstances = create_instances(FLAGS_out);
	//Melt melt;
	//melt.Test(testInstances, predictor, FLAGS_result);
	auto tester = PredictorUtils::GetTester(predictor);
	tester->Test(testInstances, predictor, FLAGS_result);
}

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
	int s = google::ParseCommandLineFlags(&argc, &argv, false);
	if (FLAGS_log_dir.empty())
		FLAGS_logtostderr = true;
	if (FLAGS_v == 0)
		FLAGS_v = FLAGS_vl;

	return RUN_ALL_TESTS();
}