#include "MLCore/TrainerFactory.h"
#include "Trainers/SVM/LinearSVM.h"
#include "Trainers/FastRank/BinaryClassificationFastRank.h"
namespace gezi {

TrainerPtr TrainerFactory::CreateTrainer(string name)
{
	boost::to_lower(name);
	if (name == "linearsvm" || name == "svm)
	{
		LOG(INFO) << "Creating LinearSVM trainer";
		return make_shared<LinearSVM>();
	}
	if (name == "fastrankbinaryclassification" ||
		name = "fastrank" || name == "gbdt")
	{
		LOG(INFO) << "Creating FastRank/GBDT trainer";
		return make_shared<BinaryClassificationFastRank>();
	}
	if (name == "kernalsvm" || name == "libsvm")
	{
		LOG(INFO) << "Creating KernalSVM trainer";
	}
	if (name == "randomforest")
	{
		LOG(INFO) << "Creating RandomForest trainer";
	}
	if (name == "logisticregression" || name == "logistic")
	{
		LOG(INFO) << "Creating LogisticRegression trainer";
	}
	if (name == "binaryneuralnetwork" || name == "neural" || name == "neuralnetwork)
	{
		LOG(INFO) << "Creating BinaryNeuralNetwork trainer";
	}
	LOG(WARNING) << name << " is not supported now, return nullptr";
	return nullptr;
}


}  //----end of namespace gezi

