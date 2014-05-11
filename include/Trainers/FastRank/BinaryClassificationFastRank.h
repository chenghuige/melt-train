/**
 *  ==============================================================================
 *
 *          \file   BinaryClassificationFastRank.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 14:32:08.893867
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef BINARY_CLASSIFICATION_FAST_RANK_H_
#define BINARY_CLASSIFICATION_FAST_RANK_H_
#include "common_util.h"
#include "FastRank.h"
#include "BinaryClassificationFastRankArguments.h"
namespace gezi {

class BinaryClassificationFastRank : public FastRank
{
public:
	BitArray TrainSetLabels;

protected:
	virtual void PrepareLabels() override
	{
		TrainSetLabels = from(TrainSet.Ratings) 
			>> select([](short a) { return (bool)(a >= _args->smallestPositive); }) 
			>> to_vector();
	}

	virtual FastRankArgumentsPtr GetArguments() override
	{
		return make_shared<BinaryClassificationFastRankArguments>();
	}

	virtual ObjectiveFunctionPtr ConstructObjFunc() override
	{
		return make_shared<BinaryClassificationObjectiveFunction>(TrainSet, TrainSetLabels, *_args);
	}

	virtual void InitializeTests() override
	{

	}
private:
};

}  //----end of namespace gezi

#endif  //----end of BINARY_CLASSIFICATION_FAST_RANK_H_
