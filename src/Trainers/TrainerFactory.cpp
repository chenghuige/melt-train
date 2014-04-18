#include "MLCore/TrainerFactory.h"
#include "Trainers/SVM/LinearSVM.h"
namespace gezi {

TrainerPtr TrainerFactory::CreateTrainer(string name_)
{
	string name = boost::to_lower_copy(name_);
	if (name == "linearsvm")
	{
		return make_shared<LinearSVM>();
	}
/*	if (name == "fastrankbinaryclassification")
	{

	}*/
	LOG(WARNING) << name_ << " is not supported now, return nullptr";
	return nullptr;
}


}  //----end of namespace gezi

