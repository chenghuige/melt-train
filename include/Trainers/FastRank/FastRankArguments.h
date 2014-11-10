/**
 *  ==============================================================================
 *
 *          \file   FastRankArguments.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-11 07:35:49.339090
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef FAST_RANK_ARGUMENTS_H_
#define FAST_RANK_ARGUMENTS_H_

namespace gezi {

struct FastRankArguments 
{
	int numTrees = 100;
	int numLeaves = 20;
	int minInstancesInLeaf = 10;
	Float learningRate = 0.2;

	bool calibrateOutput = true; //calibrate| use calibrator to gen probability?
	string calibratorName = "sigmoid"; //calibrator| sigmoid/platt naive pav @TODO 移动到Trainer父类处理

	int maxBins = 255; //mb|Maximum number of distinct values (bins) per feature
	Float sparsifyRatio = 0.3;//sr|if not big data (large instances num, large feature num can set 0 so to be all dense) that will make running faster

	unsigned randSeed = 0x7b; //rs|controls wether the expermient can reproduce, 0 means not reproduce rngSeed
	bool randomStart = false; //rst|Training starts from random ordering (determined by /r1)
	
	int histogramPoolSize = -1; //|[2, numLeaves - 1]

	int maxTreeOutput = 100; //mo|Upper bound on absolute value of single tree output

	bool affineRegressionTrees = false; //art|Use affine regression trees
	bool allowDummyRootSplits = true; //dummies|When a root split is impossible, construct a dummy empty tree rather than fail

	Float baggingTrainFraction = 0.7;//bagfrac|Percentage of training queries used in each bag
	bool bestStepRankingRegressionTrees = false; //bsr|Use best ranking step trees

	Float entropyCoefficient = 0; //e|The entropy (regularization) coefficient between 0 and 1

	int derivativesSampleRate = 1; //dsr|same each query 1 in k times in the GetDerivatives function

	Float smoothing = 0.0; //s|Smoothing paramter for tree regularization

	bool compressEnsemble = false; //cmp|Compress the tree Ensemble

	Float featureFirstUsePenalty = 0;
	Float featureReusePenalty = 0;
	Float softmaxTemperature = 0;

	Float featureFraction = 1;
	Float splitFraction = 1;
	bool filterZeroLambdas = false;
	Float gainConfidenceLevel = 0;
};

typedef shared_ptr<FastRankArguments> FastRankArgumentsPtr;
}  //----end of namespace gezi

#endif  //----end of FAST_RANK_ARGUMENTS_H_
