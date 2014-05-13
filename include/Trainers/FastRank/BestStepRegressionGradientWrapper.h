/**
 *  ==============================================================================
 *
 *          \file   BestStepRegressionGradientWrapper.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-13 12:17:04.247603
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef BEST_STEP_REGRESSION_GRADIENT_WRAPPER_H_
#define BEST_STEP_REGRESSION_GRADIENT_WRAPPER_H_
#include "IGradientAdjuster.h"
namespace gezi {

class BestStepRegressionGradientWrapper : public IGradientAdjuster
{
public:
	virtual dvec& AdjustTargetAndSetWeights(dvec& gradient, ObjectiveFunction& objFunction, dvec*& targetWeights)
	{
		targetWeights = &objFunction.Weights();
		return gradient;
	}
protected:
private:

};

}  //----end of namespace gezi

#endif  //----end of BEST_STEP_REGRESSION_GRADIENT_WRAPPER_H_
