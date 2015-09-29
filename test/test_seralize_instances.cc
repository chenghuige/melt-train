/** 
 *  ==============================================================================
 * 
 *          \file   test_seralize_instances.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2015-09-23 16:27:35.475984
 *  
 *  \Description:
 *
 *  ==============================================================================
 */

#define private public
#define protected public
#include "common_util.h"

#include "Prediction/Instances/Instance.h"
#include "Prediction/Instances/Instances.h"
#include "Prediction/Instances/HeaderSchema.h"
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
	//Instance inst;
	//ser::save(inst, "inst.bin");

	//{
	//	ListInstances insts;
	//	auto inst = make_shared<Instance>();
	//	insts.push_back(inst);
	//	ser::save(insts, "insts.bin");
	//}	

	{
		Instances insts;
		auto inst = make_shared<Instance>();
		insts.push_back(inst);
		ser::save_json(insts, "insts2.bin");
	}
	//{
	//	HeaderSchema shema;
	//	shema.numClasses = 3;
	//	serialize_util::save_json(shema, "a.bin");
	//	//{
	//	//	HeaderSchema shema;
	//	//	ser::load(shema, "a.bin");
	//	//	Pval(shema.numClasses);
	//	//}
	//}
	//{
	//	FeatureNamesVector v;
	//	ser::save(v, "v.bin");
	//}
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

		run();

	 return 0;
}
