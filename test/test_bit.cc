/** 
 *  ==============================================================================
 * 
 *          \file   test_bit.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2014-05-17 22:23:11.830949
 *  
 *  \Description:
 *
 *  ==============================================================================
 */

#define _DEBUG
#define private public
#define protected public
#include "common_util.h"

using namespace std;
using namespace gezi;
DEFINE_int32(vl, 0, "vlog level");
//DEFINE_string(i, "", "input");
//DEFINE_string(o, "", "output");
DEFINE_string(type, "simple", "");

TEST(bit, func)
{
	Pval((~0));
	Pval((~1));
	Pval((~-1));
	Pval((~-2));
	Pval((~-3));
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
