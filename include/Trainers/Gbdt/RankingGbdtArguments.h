/**
 *  ==============================================================================
 *
 *          \file   RankingGbdtArguments.h
 *
 *        \author   chenghuige
 *
 *          \date   2015-05-14 11:30:22.199097
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef RANKING_GBDT_ARGUMENTS_H_
#define RANKING_GBDT_ARGUMENTS_H_

#include "GbdtArguments.h"

namespace gezi {

struct RankingGbdtArguments : public GbdtArguments
{
	bool filterZeroLambdas = false;
};

}  //----end of namespace gezi

#endif  //----end of RANKING_GBDT_ARGUMENTS_H_
