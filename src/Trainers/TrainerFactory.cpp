#include "MLCore/TrainerFactory.h"
#include "Trainers/SVM/LinearSVM.h"
namespace gezi {

TrainerPtr TrainerFactory::CreateTrainer(string name)
{
	boost::to_lower(name);
	if (name == "linearsvm")
	{
		return make_shared<LinearSVM>();
	}
/*	if (name == "fastrankbinaryclassification")
	{

	}*/
	LOG(WARNING) << name << " is not supported now, return nullptr";
	return nullptr;
}


}  //----end of namespace gezi

