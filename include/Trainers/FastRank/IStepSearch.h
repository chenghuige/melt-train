/**
 *  ==============================================================================
 *
 *          \file   IStepSearch.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-10 22:42:23.435469
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef I_STEP_SEARCH_H_
#define I_STEP_SEARCH_H_
#include "RegressionTree.h"
#include "DocumentPartitioning.h"
#include "ScoreTracker.h"
namespace gezi {

class IStepSearch 
{
public:
	void AdjustTreeOutputs(RegressionTree& tree, DocumentPartitioning& partitioning, ScoreTracker& trainingScores) = 0;
protected:
private:

};

}  //----end of namespace gezi

#endif  //----end of I_STEP_SEARCH_H_
