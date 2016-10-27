/**
 *  ==============================================================================
 *
 *          \file   melt.cc
 *
 *        \author   chenghuige
 *
 *          \date   2014-02-02 12:36:22.355574
 *
 *  \Description:
 *  ==============================================================================
 */

#define private public
#define protected public
//#define NO_BAIDU_DEP
#include "common_util.h"
#include "Run/Melt.h"
#include "rabit_util.h"
using namespace std;
using namespace gezi;
DEFINE_int32(vl, 0, "vlog level");
DEFINE_bool(silent, false, "silent mode, will set vl = -1");
DEFINE_bool(quiet, false, "quiet mode, will set vl = -1 for node which rank is no 0");

DECLARE_string(c);

void ShowMeltHelp()
{
  cout << "\n";
  cout << " version: " << get_version() << "\n";
  cout << "	Default command is cross validation: <./melt feature.txt> will do 5 fold cross validation using LinearSVM trainer with it's default settings\n";
  fmt::print_colored(fmt::RED, "	For more commands: <./melt -c help>\n");
  fmt::print_colored(fmt::RED, "	Show supported trainers: <./melt -c helptrainers>\n");
  fmt::print_colored(fmt::RED, "	Show trainer setting try : <./melt -c helptrainer -cl linearsvm> <./melt -c helptrainer -cl gbdt>...\n");
  fmt::print_colored(fmt::RED, "	Try also like <./melt --helpmatch LinearSVM>, <./melt --helpmatch Gbdt> for trainer settings when no or less info show using -c helptrainer\n");
  cout << "	Try <./melt --helpmatch Melt> for melt common settings\n";
  cout << "	Try <./melt --helpmatch Instance> for melt input instances parser settings\n";
  cout << "	The default trainer is LinearSVM, for other trainers use -cl, eg. <./melt feature.txt -c train -cl gbdt> will train feature.txt using gbdt trainer\n";
  cout << "	For third party trainers like sofia or vw, you may use -cls to set the classifer settings, eg. <./melt feature.txt -cl sofia -cls --looptype=roc,--iterations=100000,--lambda=0.001>\n";
  cout << "	Recommed to use melt internal format(0 started index, can be sparse or dense) but you can also use libsvm format(1 started sparse format) directly\n";

}

int main(int argc, char *argv[])
{
  Rabit rabit(argc, argv);

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::SetVersionString(get_version());

  vector<string> args;
  for (int i = 0; i < argc; i++)
  {
    string arg = argv[i];
    args.push_back(arg);
    if (arg == "-help" || arg == "--help" || arg == "-h" || arg == "--h")
    {
      ShowMeltHelp();
      break;
    }
  }

  int s = google::ParseCommandLineFlags(&argc, &argv, false);

  if (FLAGS_log_dir.empty())
    FLAGS_logtostderr = true;
  if (FLAGS_silent || (FLAGS_quiet && rabit::GetRank() != 0))
    FLAGS_vl = -1;
  if (FLAGS_v == 0)
    FLAGS_v = FLAGS_vl;

  Melt melt;
  set<string> ignores = { "help", "helptrainers", "help_trainers", "helptrainer", "help_trainer", "hts", "ht" };
  if (!ignores.count(FLAGS_c)
    && !(FLAGS_c == "write_text_model" || FLAGS_c == "wtm" || FLAGS_c == "binary_model_to_text" || FLAGS_c == "bm2t")
    && !(FLAGS_c == "text_model_to_binary" || FLAGS_c == "tm2b")
    && melt.Cmd().datafile.empty())
  {
    if (s >= argc)
    {
      google::ShowUsageWithFlags(argv[0]);

      cout << "	No input data file use -i filename or just input filename after ./melt\n";
      ShowMeltHelp();
      return -1;
    }
    melt.Cmd().datafile = argv[s];
  }
  //����ǿ�˵��FLAGS_rs��ʼ������0�������0 ����Ҫ��¼-rs    hack��
  melt.Cmd().fullArguments = melt.Cmd().fullArguments.empty() ? 
    gezi::join(args) :
    format("{} -rs {}", gezi::join(args), melt.Cmd().randSeed);
  try_create_dir(melt.Cmd().resultDir);
  melt.RunExperiments();
  return 0;
}

