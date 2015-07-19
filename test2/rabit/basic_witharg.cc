/** 
 *  ==============================================================================
 * 
 *          \file   basic_witharg.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2015-06-05 21:05:49.122072
 *  
 *  \Description:
 *
 *  ==============================================================================
 */

#define private public
#define protected public
#include "common_util.h"

#include <rabit.h>
using namespace rabit;

using namespace std;
using namespace gezi;

DEFINE_int32(vl, 0, "vlog level");
DEFINE_int32(level, 0, "min log level");
DEFINE_string(type, "simple", "");
DEFINE_bool(perf,false, "");
DEFINE_int32(num, 1, "");
DEFINE_string(i, "", "input file");
DEFINE_string(o, "", "output file");

void run()
{
	const int N = 3;
	int a[N];
	
	Pval(FLAGS_num);
	for (int i = 0; i < N; ++i) {
		a[i] = rabit::GetRank() + i;
	}
	printf("@node[%d] before-allreduce: a={%d, %d, %d}\n",
		rabit::GetRank(), a[0], a[1], a[2]);
	// allreduce take max of each elements in all processes
	Allreduce<op::Max>(&a[0], N);
	printf("@node[%d] after-allreduce-max: a={%d, %d, %d}\n",
		rabit::GetRank(), a[0], a[1], a[2]);
	// second allreduce that sums everything up
	Allreduce<op::Sum>(&a[0], N);
	printf("@node[%d] after-allreduce-sum: a={%d, %d, %d}\n",
		rabit::GetRank(), a[0], a[1], a[2]);

}

int main(int argc, char *argv[])
{
		google::InitGoogleLogging(argv[0]);
		google::InstallFailureSignalHandler();
		google::SetVersionString(get_version());
		int s = google::ParseCommandLineFlags(&argc, &argv, false);
		if (FLAGS_log_dir.empty())
				FLAGS_logtostderr = true;
		FLAGS_minloglevel = FLAGS_level;
		//LogHelper::set_level(FLAGS_level);
		if (FLAGS_v == 0)
				FLAGS_v = FLAGS_vl;

		rabit::Init(argc, argv);
		run();
		rabit::Finalize();

	 return 0;
}
