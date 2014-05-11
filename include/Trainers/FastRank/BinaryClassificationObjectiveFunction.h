/**
 *  ==============================================================================
 *
 *          \file   BinaryClassificationObjectiveFunction.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-10 22:43:10.664980
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef BINARY_CLASSIFICATION_OBJECTIVE_FUNCTION_H_
#define BINARY_CLASSIFICATION_OBJECTIVE_FUNCTION_H_

#include "Dataset.h"
#include "ObjectiveFunction.h"
#include "IStepSearch.h"
#include "BinaryClassificationFastRankArguments.h"
namespace gezi {

	class BinaryClassificationObjectiveFunction : public ObjectiveFunction, public IStepSearch
	{
	private:
		int64 _nneg;
		int64 _npos;
		bool _unbalancedSets; //us|Should we use derivatives optimized for unbalanced sets
		BitArray& Labels;

	public:
		BinaryClassificationObjectiveFunction(::Dataset& trainSet, BitArray& trainSetLabels,
			BinaryClassificationFastRankArguments& args)
			: ObjectiveFunction(trainSet, args.learningRate, args.maxTreeOutput, args.derivativesSampleRate, args.bestStepRankingRegressionTrees, args.randSeed), Labels(trainSetLabels)
		{
			_unbalancedSets = args.unbalancedSets;
			if (_unbalancedSets)
			{ //@TODO
				/*BinaryClassificationTest.ComputeExampleCounts(Labels, out _npos, out _nneg);
				if ((_nneg == 0) || (_npos == 0))
				{
				throw new Exception("Only one class in training set.");
				}*/
			}
		}

		virtual void AdjustTreeOutputs(RegressionTree& tree, DocumentPartitioning& partitioning, ScoreTracker& trainingScores) override
		{
			for (int l = 0; l < tree.NumLeaves; l++)
			{
				double output = 0.0;
				if (_bestStepRankingRegressionTrees)
				{
					output = _learningRate * tree.GetOutput(l);
				}
				else
				{
					output = (_learningRate * (tree.GetOutput(l) + 1.4E-45)) / (partitioning.Mean(_weights, Dataset.SampleWeights, l, false) + 1.4E-45);
				}
				if (output > _maxTreeOutput)
				{
					output = _maxTreeOutput;
				}
				else if (output < -_maxTreeOutput)
				{
					output = -_maxTreeOutput;
				}
				tree.SetOutput(l, output);
			}
		}



	protected:
		virtual void GetGradientInOneQuery(int query, const dvec& scores) override
		{
			double sigmoidParam = _learningRate;
			double recip_npos = 1.0;
			double recip_nneg = 1.0;
			if (_unbalancedSets)
			{
				recip_npos = 1.0 / ((double)_npos);
				recip_nneg = 1.0 / ((double)_nneg);
			}
			int label = Labels[query] ? 1 : -1;
			double recip = Labels[query] ? recip_npos : recip_nneg;
			double response = ((2.0 * label) * sigmoidParam) / (1.0 + std::exp(((2.0 * label) * sigmoidParam) * scores[query]));
			double absResponse = std::abs(response);
			_gradient[query] = response * recip;
			_weights[query] = (absResponse * ((2.0 * sigmoidParam) - absResponse)) * recip;
		}

	};

}  //----end of namespace gezi

#endif  //----end of BINARY_CLASSIFICATION_OBJECTIVE_FUNCTION_H_
