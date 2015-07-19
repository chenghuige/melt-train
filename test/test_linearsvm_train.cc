/** 
 *  ==============================================================================
 * 
 *          \file   test_linearsvm_train.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2015-05-27 13:34:06.906090
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
#include "Utils/EvaluatorUtils.h"
using namespace std;
using namespace gezi;
DEFINE_int32(vl, 0, "vlog level");

DECLARE_string(i);
DECLARE_string(valid);
DECLARE_string(evaluator);


TEST(linearsvm_train, func)
{
	auto trainer = TrainerFactory::CreateTrainer("linearsvm");
	CHECK_NE((trainer == nullptr), true);
	auto instances = create_instances(FLAGS_i);

	if (FLAGS_valid.empty())
	{
		trainer->Train(instances);
	}
	else
	{
		dynamic_pointer_cast<ValidatingTrainer>(trainer)->Train(instances,
			vector < Instances > {create_instances(FLAGS_valid)},
			EvaluatorUtils::CreateEvaluators(FLAGS_evaluator));
	}
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
