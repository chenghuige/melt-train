/**
 *  ==============================================================================
 *
 *          \file   Trainers/LibSVMTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 05:15:42.677732
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__LIB_S_V_M_TRAINER_H_
#define TRAINERS__LIB_S_V_M_TRAINER_H_

#include "ThirdTrainer.h"

class svm_problem;
namespace gezi {

class LibSVMTrainer : public ThirdTrainer
{
public:

	LibSVMTrainer()
	{
		_classiferSettings = "-b 1";
	}

	virtual string GetPredictorName() override
	{
		return "LibSVM";
	}

protected:
	virtual Float Margin(InstancePtr instance) override;

	virtual PredictorPtr CreatePredictor() override;

	virtual void ShowHelp() override;

	svm_problem  Instances2SvmProblem(Instances& instances);
	virtual void Initialize(Instances& instances) override;
	virtual void InnerTrain(Instances& instances) override;
	virtual void Finalize_(Instances& instances) override;

private:

};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__LIB_S_V_M_TRAINER_H_
