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
DECLARE_int32(iter);
DEFINE_int32(ntree, 100, "numTrees: Number of trees/iteraiton number");
DECLARE_double(lr);
DEFINE_int32(nl, 20, "numLeaves: Number of leaves maximam allowed in each regression tree");
DEFINE_int32(mil, 10, "minInstancesInLeaf: Minimal instances in leaves allowd");
namespace gezi {

	void FastRank::ParseArgs()
	{
		_args = GetArguments();
	
		if (!are_same(FLAGS_lr, 0.001)) //@TODO double 判断相同 判断是否是0 另外如果用户再输入0.0001不起作用了就 不符合逻辑了
			_args->learningRate = FLAGS_lr;
		
		if (FLAGS_iter != 50000)
			_args->numTrees = FLAGS_iter;
		else
			_args->numTrees = FLAGS_ntree;

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
