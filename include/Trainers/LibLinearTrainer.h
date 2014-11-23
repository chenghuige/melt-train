/**
 *  ==============================================================================
 *
 *          \file   Trainers/LibLinearTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 05:15:53.320426
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__LIB_LINEAR_TRAINER_H_
#define TRAINERS__LIB_LINEAR_TRAINER_H_

#include "ThirdTrainer.h"

class problem;
namespace gezi {

class LibLinearTrainer : public ThirdTrainer
{
public:
	virtual string GetPredictorName() override
	{
		return "LibLinear";
	}
	
protected:
	virtual void ShowHelp() override;

	problem Instances2problem(Instances& instances);
	virtual void Initialize(Instances& instances) override;
	virtual void InnerTrain(Instances& instances) override;
	virtual void Finalize_(Instances& instances) override;
private:
};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__LIB_LINEAR_TRAINER_H_
