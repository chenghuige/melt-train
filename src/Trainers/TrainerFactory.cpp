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

	enum class TrainerType
	{
		Unknown,
		Random,
		LinearSVM,
		BaseLineLinearSVM,
		BinaryClassificationFastRank,
		LogisticRegression,
		RandomForest,
		DecisionTree,
		KernalSVM,
		BinaryNeuralNetwork
	};

	map<string, TrainerType> _trainerTypes
	{
		{ "random", TrainerType::Random },
		{ "baselinelinearsvm", TrainerType::BaseLineLinearSVM },
		{ "baselinesvm", TrainerType::BaseLineLinearSVM },
		{ "linearsvm", TrainerType::LinearSVM },
		{ "svm", TrainerType::LinearSVM },
		{ "fastrank", TrainerType::BinaryClassificationFastRank },
		{ "gbdt", TrainerType::BinaryClassificationFastRank },
		{ "fr", TrainerType::BinaryClassificationFastRank },
	};

	TrainerPtr TrainerFactory::CreateTrainer(string name)
	{
		boost::to_lower(name);
		TrainerType trainerType = _trainerTypes[name];
	
		switch (trainerType)
		{
		case TrainerType::Random:
			VLOG(0) << "Creating Random trainer, just for test auc will be around 0.5";
			return make_shared<RandomTrainer>();
			break;
		case TrainerType::BaseLineLinearSVM:
			VLOG(0) << "Creating BaselineLinearSVM trainer, this one is slow, try use LinearSVM";
			return make_shared<BaseLineLinearSVM>();
			break;
		case TrainerType::LinearSVM:
			VLOG(0) << "Creating LinearSVM trainer";
			return make_shared<LinearSVM>();
			break;
		case TrainerType::BinaryClassificationFastRank:
			VLOG(0) << "Creating FastRank/GBDT trainer";
			return make_shared<BinaryClassificationFastRank>();
			break;
		case TrainerType::KernalSVM:
			VLOG(0) << "Creating KernalSVM trainer";
			break;
		case  TrainerType::DecisionTree:
			VLOG(0) << "Creating DecisionTree trainer";
			break;
		case TrainerType::RandomForest:
			VLOG(0) << "Creating RandomForest trainer";
			break;
		case  TrainerType::LogisticRegression:
			VLOG(0) << "Creating LogisticRegression trainer";
			break;
		case  TrainerType::BinaryNeuralNetwork:
			VLOG(0) << "Creating BinaryNeuralNetwork trainer";
			break;
		case  TrainerType::Unknown:
			break;
		default:
			break;
		}

		LOG(WARNING) << name << " is not supported now, return nullptr";
		return nullptr;
	}


}  //----end of namespace gezi

