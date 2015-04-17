/**
 *  ==============================================================================
 *
 *          \file   BaggingProvider.h
 *
 *        \author   chenghuige
 *
 *          \date   2015-04-17 20:28:11.041061
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef BAGGING_PROVIDER_H_
#define BAGGING_PROVIDER_H_

#include "Dataset.h"
#include "DocumentPartitioning.h"
#include "random_util.h"
namespace gezi {

	class BaggingProvider
	{
	public:
		Random _rand;
		const Dataset& _completeTrainerSet;
		int _maxLeaves;
		double _trainFraction;

		BaggingProvider(const Dataset& completeTrainerSet, int randomSeed, int maxLeaves, double trainFraction)
			:_completeTrainerSet(completeTrainerSet), _rand(randomSeed), _maxLeaves(maxLeaves), _trainFraction(trainFraction)
		{

		}

		void GenPartion(DocumentPartitioning& currentTrainPartition, DocumentPartitioning& currentOutOfBagPartition)
		{
			ivec trainDocs(_completeTrainerSet.NumDocs, -1);
			ivec outOfBagDocs(_completeTrainerSet.NumDocs, -1);
			int trainSize = 0;
			int outOfBagSize = 0;
			for (int i = 0; i < _completeTrainerSet.NumDocs; i++)
			{
				if (_rand.NextDouble() < _trainFraction)
				{
					trainDocs[trainSize++] = i;
				}
				else
				{
					outOfBagDocs[outOfBagSize++] = i;
				}
			}
			Pval2_2(trainSize, outOfBagSize);
			currentTrainPartition = DocumentPartitioning(trainDocs, trainSize, _maxLeaves);
			currentOutOfBagPartition = DocumentPartitioning(outOfBagDocs, outOfBagSize, _maxLeaves);
			currentTrainPartition.Initialize();
			currentOutOfBagPartition.Initialize();
		}
	protected:
	private:
		//Dataset& _completeTrainingSet;
		//DocumentPartitioning _currentOutOfBagPartition;
		//DocumentPartitioning _currentTrainPartition;
		//Random _rndGenerator;

	};

}  //----end of namespace gezi

#endif  //----end of BAGGING_PROVIDER_H_
