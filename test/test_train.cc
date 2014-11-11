/** 
 *  ==============================================================================
 * 
 *          \file   test_train.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2014-11-09 15:07:25.516188
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
using namespace std;
using namespace gezi;
DEFINE_int32(vl, 0, "vlog level");
DEFINE_string(in, "./data/feature.normed.libsvm", "input file");
DECLARE_string(cl);
DECLARE_bool(calibrate);
DECLARE_bool(norm);

TEST(train, func)
{
	FLAGS_norm = false;
	FLAGS_calibrate = false;
	auto trainer = TrainerFactory::CreateTrainer(FLAGS_cl);
	CHECK_NE((trainer == nullptr), true);
	auto instances = create_instances(FLAGS_in);
	AutoTimer timer("Train");
	trainer->Train(instances);
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