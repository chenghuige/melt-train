/** 
 *  ==============================================================================
 * 
 *          \file   test_fastrank_train.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2014-05-11 20:48:20.312025
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
using namespace std;
using namespace gezi;
DEFINE_int32(level, 0, "min log level");

TEST(fastrank_train, func)
{
	auto trainer = TrainerFactory::CreateTrainer("fastrank");
	CHECK_NE((trainer == nullptr), true);
}

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
	int s = google::ParseCommandLineFlags(&argc, &argv, false);
	if (FLAGS_log_dir.empty())
		FLAGS_logtostderr = true;
	FLAGS_minloglevel = FLAGS_level;
	//boost::progress_timer timer;
	
	return RUN_ALL_TESTS();
}
