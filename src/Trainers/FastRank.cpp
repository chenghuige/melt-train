/**
 *  ==============================================================================
 *
 *          \file   FastRank.cpp
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-12 10:09:15.775908
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef FAST_RANK_CPP_
#define FAST_RANK_CPP_

#include "common_util.h"
#include "Trainers/FastRank/FastRank.h"

DECLARE_string(calibrator);
DECLARE_uint64(rs);


DEFINE_int32(iter, 50000, "numIterExamples: Number of iterations");
DEFINE_double(lr, 0.001, "lambda: learning rate");

DEFINE_int32(nt, 100, "numTrees: Number of trees/iteraiton number");
DECLARE_double(lr);
DEFINE_int32(nl, 20, "numLeaves: Number of leaves maximam allowed in each regression tree");
DEFINE_int32(mil, 10, "minInstancesInLeaf: Minimal instances in leaves allowd");
namespace gezi {

	virtual void FastRank::ParseArgs()
	{
		_args = GetArguments();
	
		if (!are_same(FLAGS_lr, 0.001)) //@TODO double 判断相同 判断是否是0
			_args->learningRate = FLAGS_lr;
		
		if (FLAGS_iter != 50000)
			_args->numTrees = FLAGS_iter;
		else
			_args->numTrees = FLAGS_nt;

		_args->numLeaves = FLAGS_nl;
		_args->minInstancesInLeaf = FLAGS_mil;

		//---- doing more
		if (_args->histogramPoolSize < 2)
		{
			_args->histogramPoolSize = (_args->numLeaves * 2) / 3;
		}
		if (_args->histogramPoolSize >(_args->numLeaves - 1))
		{
			_args->histogramPoolSize = _args->numLeaves - 1;
		}
	}
}

#endif  //----end of FAST_RANK_CPP_
