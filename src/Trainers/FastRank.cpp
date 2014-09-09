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

DECLARE_bool(calibrate);
DECLARE_string(calibrator);

DECLARE_uint64(rs);
DECLARE_int32(iter);
DEFINE_int32(ntree, 100, "numTrees: Number of trees/iteraiton number");
DECLARE_double(lr);
DEFINE_int32(nl, 20, "numLeaves: Number of leaves maximam allowed in each regression tree");
DEFINE_int32(mil, 10, "minInstancesInLeaf: Minimal instances in leaves allowd");
DEFINE_bool(bsr, false, "bestStepRankingRegressionTrees: ");
DEFINE_double(sp, 0.1, "Sparsity level needed to use sparse feature representation, if 0.3 means be sparsify only if real data less then 30%, 0-1 the smaller more dense and faster but use more memeory");
DEFINE_double(ff, 1, "The fraction of features (chosen randomly) to use on each iteration");
DEFINE_int32(mb, 255, "Maximum number of distinct values (bins) per feature");
DEFINE_int32(ps, -1, "The number of histograms in the pool (between 2 and numLeaves)");

namespace gezi {

	void FastRank::ParseArgs()
	{
		_args = GetArguments();

		_args->calibrateOutput = FLAGS_calibrate; //@TODO to Trainer deal
		_args->calibratorName = FLAGS_calibrator;

		if (!are_same(FLAGS_lr, 0.001)) //@TODO double �ж���ͬ �ж��Ƿ���0 ��������û�������0.0001���������˾� �������߼���
			_args->learningRate = FLAGS_lr;

		if (FLAGS_iter != 50000)
			_args->numTrees = FLAGS_iter;
		else
			_args->numTrees = FLAGS_ntree;

		if (FLAGS_nl < 2)
			LOG(WARNING) << "The number of leaves must be >= 2, so use default value instead";
		else
			_args->numLeaves = FLAGS_nl;

		_args->randSeed = FLAGS_rs;

		_args->minInstancesInLeaf = FLAGS_mil;

		_args->bestStepRankingRegressionTrees = FLAGS_bsr;
		_args->sparsifyRatio = FLAGS_sp;

		_args->featureFraction = FLAGS_ff;

		_args->maxBins = FLAGS_mb;

		//---- doing more ����о����Ƕ���historygram����pool�洢�� Ϊʲô���� *2/3��������һ�㲻����
		//�����numLeaves - 1��ô���еķ�root���� ������parent�洢�� ����substract ��Ҫ���ڴ濼�ǰɣ�
		_args->histogramPoolSize = FLAGS_ps;
		if (_args->histogramPoolSize < 2)
		{
			_args->histogramPoolSize = (_args->numLeaves * 2) / 3;
		}
		if (_args->histogramPoolSize > (_args->numLeaves - 1))
		{
			_args->histogramPoolSize = _args->numLeaves - 1;
		}
	}
}

#endif  //----end of FAST_RANK_CPP_
