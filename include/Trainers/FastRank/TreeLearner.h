/**
 *  ==============================================================================
 *
 *          \file   TreeLearner.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-08 19:12:50.873604
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TREE_LEARNER_H_
#define TREE_LEARNER_H_

#include "common_def.h"
#include "Dataset.h"
#include "DocumentPartitioning.h"
namespace gezi {

	class TreeLearner
	{
	public:
		int NumLeaves;
		DocumentPartitioning Partitioning;
		Dataset& TrainData;
		TreeLearner(Dataset& trainData, int numLeaves)
			:TrainData(trainData), NumLeaves(numLeaves), Partitioning(trainData.NumDocs, numLeaves)
		{
		}

		virtual RegressionTree FitTargets(dvec& targets) = 0;

		static string TargetWeightsDatasetName()
		{
			return "TargetWeightsDataset";
		}

		bool IsFeatureOk(int index)
		{
			return TrainData.Features[index].NumBins() > 1;
		}
	protected:
	private:

	};

	typedef shared_ptr<TreeLearner> TreeLearnerPtr;

}  //----end of namespace gezi

#endif  //----end of TREE_LEARNER_H_
