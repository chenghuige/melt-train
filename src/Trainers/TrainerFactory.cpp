#include "MLCore/TrainerFactory.h"
#include "Trainers/SVM/LinearSVM.h"
#include "Trainers/FastRank/BinaryClassificationFastRank.h"
#include "Trainers/RandomTrainer.h"
namespace gezi {

TrainerPtr TrainerFactory::CreateTrainer(string name)
{
	boost::to_lower(name);
	if (name == "linearsvm" || name == "svm")
	{
		VLOG(0) << "Creating LinearSVM trainer";
		return make_shared<LinearSVM>();
	}
	if (name == "fastrankbinaryclassification" ||
		name == "fastrank" || name == "gbdt" || name == "fr")
	{
		VLOG(0) << "Creating FastRank/GBDT trainer";
		return make_shared<BinaryClassificationFastRank>();
	}
	if (name == "kernalsvm" || name == "libsvm")
	{
		VLOG(0) << "Creating KernalSVM trainer";
	}
	if (name == "randomforest")
	{
		VLOG(0) << "Creating RandomForest trainer";
	}
	if (name == "logisticregression" || name == "logistic")
	{
		VLOG(0) << "Creating LogisticRegression trainer";
	}
	if (name == "binaryneuralnetwork" || name == "neural" || name == "neuralnetwork")
	{
		VLOG(0) << "Creating BinaryNeuralNetwork trainer";
	}
	//just for test
	if (name == "random")
	{
		VLOG(0) << "Creatring random trainer(do nothing will genearate a random predictor to predict 0,1 randomly)";
		return make_shared<RandomTrainer>();
	}
	
	LOG(WARNING) << name << " is not supported now, return nullptr";
	return nullptr;
}


}  //----end of namespace gezi

