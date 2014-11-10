/**
 *  ==============================================================================
 *
 *          \file   QueryWeightsBestResressionStepGradientWrapper.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-07-30 10:50:02.892042
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef QUERY_WEIGHTS_BEST_RESRESSION_STEP_GRADIENT_WRAPPER_H_
#define QUERY_WEIGHTS_BEST_RESRESSION_STEP_GRADIENT_WRAPPER_H_

#include "IGradientAdjuster.h"
namespace gezi {

	class QueryWeightsBestStepRegressionGradientWrapper : public IGradientAdjuster
	{
	public:
		virtual Fvec& AdjustTargetAndSetWeights(Fvec& gradient, ObjectiveFunction& objFunction, Fvec*& targetWeights)
		{
			Fvec& sampleWeights = objFunction.Dataset.SampleWeights;
			//@TODO �����ȷ�� ���� �ɴ� �� ������һ������ Fvec weights�أ�
			shared_ptr<Fvec> pweights = make_shared<Fvec>(gradient.size());
			Fvec& weights = *pweights;
			for (size_t i = 0; i < gradient.size(); i++)
			{
				Float queryWeight = sampleWeights[i];
				gradient[i] = gradient[i] * queryWeight;
				weights[i] = objFunction.Weights()[i] * queryWeight;
			}
			targetWeights = pweights.get();
			return gradient;
		}
	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of QUERY_WEIGHTS_BEST_RESRESSION_STEP_GRADIENT_WRAPPER_H_
