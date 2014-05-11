/** 
 *  ==============================================================================
 * 
 *          \file   test_fastrank_instances2dataset.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2014-05-08 11:09:22.252381
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
#include "Trainers/FastRank/InstancesToDataset.h"
#include "Trainers/FastRank/RegressionTree.h"
using namespace std;
using namespace gezi;
DEFINE_int32(level, 0, "min log level");
DEFINE_string(in, "./data/feature.txt", "input");
//DEFINE_string(o, "", "output");
DEFINE_string(type, "simple", "");

#include "Trainers/FastRank/LeastSquaresRegressionTreeLearner.h"
#include "Trainers/FastRank/Feature.h"
#include "Trainers/FastRank/FeatureHistogram.h"
#include "Trainers/FastRank/FastRank.h"
#include "Trainers/FastRank/BinaryClassificationFastRank.h"

TEST(fastrank_instances2dataset, func)
{
	/*Noticer nt("instances2dataset");
	auto instances = create_instances(FLAGS_in);
	instances.PrintSummary();
	auto dataSet = InstancesToDataset::Convert(instances);*/
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