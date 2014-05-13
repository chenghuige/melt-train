/**
 *  ==============================================================================
 *
 *          \file   TrivialGradientWrapper.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-10 22:34:23.300925
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRIVIAL_GRADIENT_WRAPPER_H_
#define TRIVIAL_GRADIENT_WRAPPER_H_
#include "IGradientAdjuster.h"
namespace gezi {
	class TrivialGradientWrapper : public IGradientAdjuster
	{
	public:
		virtual dvec& AdjustTargetAndSetWeights(dvec& gradient, ObjectiveFunction& objFunction, dvec*& targetWeights)
		{
			targetWeights = NULL;
			return gradient;
		}

		string TargetWeightsSetName()
		{
			return "";
		}
	};

}  //----end of namespace gezi

#endif  //----end of TRIVIAL_GRADIENT_WRAPPER_H_
