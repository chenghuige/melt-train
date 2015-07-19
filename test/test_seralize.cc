/** 
 *  ==============================================================================
 * 
 *          \file   test_seralize.cc
 *
 *        \author   chenghuige   
 *
 *          \date   2015-06-05 22:25:39.926562
 *  
 *  \Description:
 *
 *  ==============================================================================
 */

#define private public
#define protected public
#include "common_util.h"
#include "melt_common.h"
using namespace std;
using namespace gezi;

DEFINE_int32(vl, 0, "vlog level");
DEFINE_int32(level, 0, "min log level");
DECLARE_string(i);
DECLARE_string(cl);

void run()
{
	auto trainer = TrainerFactory::CreateTrainer(FLAGS_cl);
	auto instances = create_instances(FLAGS_i);
	trainer->Train(instances);
	auto predictor = trainer->CreatePredictor();
	string s = serialize_util::save(predictor);
	Pval(s);
	auto predictor2 = serialize_util::load<PredictorPtr>(s);
	Pval(predictor2->Name());
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
