/**
 *  ==============================================================================
 *
 *          \file   RankingObjectiveFunction.h
 *
 *        \author   chenghuige
 *
 *          \date   2016-06-19 17:41:32.376738
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef GEZI_RANKING_OBJECTIVE_FUNCTION_H_
#define GEZI_RANKING_OBJECTIVE_FUNCTION_H_

#include "Dataset.h"
#include "ObjectiveFunction.h"
#include "IStepSearch.h"
#include "RankingGbdtArguments.h"

namespace gezi {

	class RankingObjectiveFunction
	{
	protected:
		Fvec& Labels;

	public:
		RankingObjectiveFunction(Fvec& trainSetLabels)
			:Labels(trainSetLabels)
		{

		}
	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of GEZI_RANKING_OBJECTIVE_FUNCTION_H_
