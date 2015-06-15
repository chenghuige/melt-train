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
#include "Trainers/FastRank/BinaryClassificationFastRank.h"

DECLARE_bool(calibrate);
DECLARE_string(calibrator);
DECLARE_uint64(numCali);

DECLARE_uint64(rs);
DECLARE_int32(iter);
DEFINE_int32(ntree, 100, "numTrees: Number of trees/iteraiton number");
DECLARE_double(lr);
DEFINE_int32(nl, 20, "numLeaves: Number of leaves maximam allowed in each regression tree");
DEFINE_int32(mil, 10, "minInstancesInLeaf: Minimal instances in leaves allowd");
DEFINE_bool(bsr, false, "bestStepRankingRegressionTrees: @TODO");
DEFINE_double(sp, 0.1, "Sparsity level needed to use sparse feature representation, if 0.3 means be sparsify only if real data less then 30%, 0-1 the smaller more dense and faster but use more memeory");
DEFINE_double(ff, 1, "The fraction of features (chosen randomly) to use on each iteration");
DEFINE_double(sf, 1, "The fraction of features (chosen randomly) to use on each split");
DEFINE_int32(mb, 255, "Maximum number of distinct values (bins) per feature");
DEFINE_int32(ps, -1, "The number of histograms in the pool (between 2 and numLeaves - 1)");
DEFINE_bool(psc, false, "Wether first randomly select a subset of features and then pick the feature that maximizes gain or post do this: @FIXME");

DEFINE_int32(bag, 0, "Number of trees in each bag (0 for disabling bagging)");
DEFINE_double(bagfrac, 0.7, "Percentage of training queries used in each bag");
//bagging Ӧ�û��������⡣���� �ؼ���TrainSet�����⣿ NumDocs �ȵ� scores�ȵ�
DEFINE_int32(nbag, 1, "NumBags|if nbag > 1 then we actually has nbag * numtress = totalTrees  @FIXME"); 
DEFINE_double(nbagfrac, 0.7, "Percentage of training queries used in each bag");
DEFINE_bool(bstrap, false, "BootStrap|wether to use bootstrap full sampling with replacement or each sampling use bagfrac");
DEFINE_double(bsfrac, 1.0, "BootStrapFraction|traditional bootstrap sampling will use all data");

DEFINE_double(entropy, 0, "entropyCoefficient|sets the entropy coefficient, which encourages the algorithm to prefer balanced splits in the tree (splits where an equal number of training documents go in each direction)");

DEFINE_bool(rstart, false, "randomStart|Initialize with one random tree");

DEFINE_int32(maxfs, 0, "maxFeaturesShow| max print feature num");

namespace gezi {

	void FastRank::ParseArgs()
	{
		_args = CreateArguments();

		_args->calibrateOutput = FLAGS_calibrate; //@TODO to Trainer deal
		_args->calibratorName = FLAGS_calibrator;

		if (!are_same(FLAGS_lr, 0.001)) //@TODO Float �ж���ͬ �ж��Ƿ���0 ��������û�������0.0001���������˾� �������߼���
			_args->learningRate = FLAGS_lr;

		if (FLAGS_iter != 50000) //����LinearSVM���ֶ���Ĳ���iter,��Ĭ��ֵ��50000
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
		_args->splitFraction = FLAGS_sf;
		_args->preSplitCheck = FLAGS_psc;

		_args->baggingSize = FLAGS_bag;
		_args->baggingTrainFraction = FLAGS_bagfrac;
		_args->numBags = FLAGS_nbag;
		_args->nbaggingTrainFraction = FLAGS_nbagfrac;

		_args->boostStrap = FLAGS_bstrap;
		_args->bootStrapFraction = FLAGS_bsfrac;

		if (_args->baggingSize != 0)
		{
			CHECK_EQ(_args->numTrees % _args->baggingSize, 0) << "numTrees must be n * baggingSize";
		}

		_args->entropyCoefficient = FLAGS_entropy;

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

		_args->randomStart = FLAGS_rstart;

		_args->maxFeaturesShow = FLAGS_maxfs;
	}

	void BinaryClassificationFastRank::ParseClassificationArgs()
	{
		_args->maxCalibrationExamples = FLAGS_numCali;
	}
}

#endif  //----end of FAST_RANK_CPP_
