/**
 *  ==============================================================================
 *
 *          \file   openmp/hello.cc
 *
 *        \author   chenghuige
 *
 *          \date   2014-02-12 17:24:14.319541
 *
 *  \Description:
 *
 *  ==============================================================================
 */

#define private public
#define protected public
#include "common_util.h"
#include <omp.h>

using namespace std;
DEFINE_int32(level, 0, "min log level");
DEFINE_string(type, "simple", "");
DEFINE_bool(perf, false, "");
DEFINE_int32(num, 1, "");
DEFINE_string(i, "", "input file");
DEFINE_string(o, "", "output file");

void run()
{
	Pval(omp_get_num_threads());
	Pval(gezi::get_num_threads());
#pragma omp parallel 
	cout << "haha\n";
}

int main(int argc, char *argv[])
{
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
	int s = google::ParseCommandLineFlags(&argc, &argv, false);
	if (FLAGS_log_dir.empty())
		FLAGS_logtostderr = true;
	FLAGS_minloglevel = FLAGS_level;
	run();

	return 0;
}
