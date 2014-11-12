#include "MLCore/TrainerFactory.h"
#include "Trainers/SVM/BaseLineLinearSVM.h"
#include "Trainers/SVM/LinearSVM.h"
#include "Trainers/FastRank/BinaryClassificationFastRank.h"
#include "Trainers/RandomTrainer.h"
namespace gezi {

	void TrainerFactory::PrintTrainersInfo()
	{
		VLOG(0) << "BinaryClassification Trainers";
		VLOG(0) << "[LinearSVM] -cl linearsvm or svm | ./melt -helpmatch LinearSVM.cpp";
		VLOG(0) << "super fast, for classification with large number of features like text classification";
		VLOG(0) << "[FastRank] -cl fastrank or fr or gbdt | ./melt -helpmatch FastRank.cpp";
		VLOG(0) << "fast, best auc result for most classification problems with num features < 10^5";
		VLOG(0) << "For per trainer parameters use, like LinearSVM just <./melt -helpmatch LinearSVM.cpp>, for other common parameters <./melt -helpmatch Melt>";
		VLOG(0) << "The default trainer is LinearSVM, if use other trainers use -cl, eg. <./melt feature.txt -c train -cl gbdt> will train feature.txt using gbdt trainer";
	}

	TrainerPtr TrainerFactory::CreateTrainer(string name)
	{
		boost::to_lower(name);
		if (name == "baselinelinearsvm" || name == "baselinesvm")
		{
			VLOG(0) << "Creating BaselineLinearSVM trainer";
			return make_shared<BaseLineLinearSVM>();
		}
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

