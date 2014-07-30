/**
 *  ==============================================================================
 *
 *          \file   QueryWeightsGradientWrapper.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-07-30 10:50:18.026823
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef QUERY_WEIGHTS_GRADIENT_WRAPPER_H_
#define QUERY_WEIGHTS_GRADIENT_WRAPPER_H_

#include "IGradientAdjuster.h"
namespace gezi {

	class QueryWeightsGradientWrapper : public IGradientAdjuster
	{
	public:
		virtual dvec& AdjustTargetAndSetWeights(dvec& gradient, ObjectiveFunction& objFunction, dvec*& targetWeights)
		{
			dvec& sampleWeights = objFunction.Dataset.SampleWeights;
			for (size_t i = 0; i < gradient.size(); i++)
			{ //@TODO 修改了gradient并返回 修改会有其它影响吗
				gradient[i] = gradient[i] * sampleWeights[i];
			}
			targetWeights = &sampleWeights;
			return gradient;
		}
	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of QUERY_WEIGHTS_GRADIENT_WRAPPER_H_
