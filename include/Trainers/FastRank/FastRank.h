/**
 *  ==============================================================================
 *
 *          \file   Trainers/FastRank/FastRank.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-07 16:09:21.545812
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__FAST_RANK__FAST_RANK_H_
#define TRAINERS__FAST_RANK__FAST_RANK_H_
#include "common_def.h"
#include "MLCore/Trainer.h"
#include "Dataset.h"
#include "Ensemble.h"
#include "OptimizationAlgorithm.h"
#include "FastRankArguments.h"
namespace gezi {

	class FastRank : public Trainer
	{
	public:
		Dataset TrainSet;

	/*	enum class OptimizationAlgorithm
		{
			GradientDescent,
			AcceleratedGradientDescent,
			ConjugateGradientDescent
		}*/

		struct Arguments
		{
			//important
		
		};

		FastRank()
		{
			ParseArgs();
		}

		void ParseArgs()
		{
			if (_args.histogramPoolSize < 2)
			{
				_args.histogramPoolSize = (_args.numLeaves * 2) / 3;
			}
			if (_args.histogramPoolSize >(_args.numLeaves - 1))
			{
				_args.histogramPoolSize = _args.numLeaves - 1;
			}
		}

		virtual void InnerTrain(Instances& instances) override
		{
			TrainSet = InstancesToDataset::Convert(instances, _args.maxBins, _args.sparsifyRatio);


		}
	protected:
	private:
		Ensemble _ensemble;
		shared_ptr<OptimizationAlgorithm> _optimizationAlgorithm = nullptr;

		dmat InitTestScores;
		dvec InitTrainScores;
		dvec InitValidScores;

		FastRankArguments _args;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__FAST_RANK_H_
