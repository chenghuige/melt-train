/**
 *  ==============================================================================
 *
 *          \file   IGradientAdjuster.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-10 22:33:46.950848
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef I_GRADIENT_ADJUSTER_H_
#define I_GRADIENT_ADJUSTER_H_
#include "ObjectiveFunction.h"
namespace gezi {

class IGradientAdjuster 
{
public:
	dvec AdjustTargetAndSetWeights(dvec& gradient, ObjectiveFunction& objFunction, dvec& targetWeights) = 0;
protected:
private:

};

typedef shared_ptr<IGradientAdjuster> IGradientAdjusterPtr;
}  //----end of namespace gezi

#endif  //----end of I_GRADIENT_ADJUSTER_H_
