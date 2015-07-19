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
#include "Trainers/FastRank/InstancesToDataset.h"
#include "Prediction/Instances/instances_util.h"
using namespace std;
using namespace gezi;
DEFINE_int32(vl, 0, "vlog level");
DEFINE_string(type, "simple", "");
DEFINE_int32(col, 183, "");

DECLARE_string(i);
#include "Trainers/FastRank/TreeLearner.h"
//struct TreeLearner2 : public TreeLearner
//{
//	TreeLearner2(Dataset& trainData, int numLeaves)
//		:TreeLearner(trainData, numLeaves)
//	{
//
//	}
//	RegressionTree FitTargets(Fvec& targets)
//	{
//		return RegressionTree(20);
//	}
//};

TEST(fastrank_instances2dataset, func)
{
	Noticer nt("instances2dataset");
	auto instances = create_instances(FLAGS_i);
	instances.PrintSummary();
	auto dataSet = InstancesToDataset::Convert(instances);
	Pvec(dataSet.Features[FLAGS_col].BinMedians);
	Pvec(dataSet.Features[FLAGS_col].BinUpperBounds);
	Pval(dataSet.NumDocs);
	//TreeLearnerPtr learner = make_shared<TreeLearner2>(dataSet, 20);
	//TreeLearner2 learner(dataSet, 20);
	{
		dvec vec;
		for (size_t i = 0; i < instances.size(); i++)
		{
			vec.push_back(instances(i, 183));
		}
		Fvec upperBounds, medians;
		find_bins(vec, 255, upperBounds, medians);
		Pvec(medians);
		Pvec(upperBounds);
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
