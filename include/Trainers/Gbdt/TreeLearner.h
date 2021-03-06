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

		Fvec* TargetWeights = NULL;

		TreeLearner(Dataset& trainData, int numLeaves)
			:TrainData(trainData), NumLeaves(numLeaves), Partitioning(trainData.NumDocs, numLeaves)
		{
		}

		//@TODO check if targets can be const
		virtual RegressionTree FitTargets(const BitArray& activeFeatures, Fvec& targets) = 0;

		static string TargetWeightsDatasetName()
		{
			return "TargetWeightsDataset";
		}

		virtual bool IsFeatureOk(int index)
		{
			//return TrainData.Features[index].NumBins() > 1;
			return (*_activeFeatures)[index];
		}
	protected:
		BitArray* _activeFeatures;
	private:
	};

	typedef shared_ptr<TreeLearner> TreeLearnerPtr;

}  //----end of namespace gezi

#endif  //----end of TREE_LEARNER_H_
