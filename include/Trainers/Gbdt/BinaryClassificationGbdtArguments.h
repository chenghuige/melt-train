/**
 *  ==============================================================================
 *
 *          \file   BinaryClassificationGbdtArguments.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-11 07:36:10.199181
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef BINARY_CLASSIFICATION_GBDT_ARGUMENTS_H_
#define BINARY_CLASSIFICATION_GBDT_ARGUMENTS_H_
#include "GbdtArguments.h"
namespace gezi {

struct BinaryClassificationGbdtArguments : public GbdtArguments
{
	short smallestPositive = 1; //pos|The smallest HRS label that maps to a positive (default: 1)
	bool unbalancedSets = false; //us|Should we use derivatives optimized for unbalanced sets
	int maxCalibrationExamples = 1000000;
};

}  //----end of namespace gezi

#endif  //----end of BINARY_CLASSIFICATION_GBDT_ARGUMENTS_H_
