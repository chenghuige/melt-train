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
namespace gezi {

class LibLinearTrainer : public ThirdTrainer
{
public:
	virtual string GetPredictorName() override
	{
		return "LibLinear";
	}

protected:
private:

};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__LIB_LINEAR_TRAINER_H_
