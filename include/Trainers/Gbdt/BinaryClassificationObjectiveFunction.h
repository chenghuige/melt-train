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
#include "BinaryClassificationGbdtArguments.h"
namespace gezi {

	//class BinaryClassificationObjectiveFunction : public ObjectiveFunction, public IStepSearch
	class BinaryClassificationObjectiveFunction : public ObjectiveFunctionImpl<BinaryClassificationObjectiveFunction>, public IStepSearch
	{
	private:
		int64 _nneg;
		int64 _npos;
		bool _unbalancedSets; //us|Should we use derivatives optimized for unbalanced sets
		BitArray& Labels;

	public:
		BinaryClassificationObjectiveFunction(gezi::Dataset& trainSet, BitArray& trainSetLabels,
			BinaryClassificationGbdtArguments& args)
			: ObjectiveFunctionImpl(trainSet, args.learningRate, args.maxTreeOutput, args.derivativesSampleRate, args.bestStepRankingRegressionTrees, args.randSeed), Labels(trainSetLabels)
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

			//GetGradientInOneQuery = [this](int query, const Fvec& scores)
			//{
			//	Float sigmoidParam = _learningRate;
			//	Float recip_npos = 1.0;
			//	Float recip_nneg = 1.0;
			//	if (_unbalancedSets)
			//	{
			//		recip_npos = 1.0 / ((Float)_npos);
			//		recip_nneg = 1.0 / ((Float)_nneg);
			//	}
			//	int label = Labels[query] ? 1 : -1;
			//	Float recip = Labels[query] ? recip_npos : recip_nneg;
			//	Float response = ((2.0 * label) * sigmoidParam) / (1.0 + std::exp(((2.0 * label) * sigmoidParam) * scores[query]));
			//	Float absResponse = std::abs(response);
			//	_gradient[query] = response * recip;
			//	_weights[query] = (absResponse * ((2.0 * sigmoidParam) - absResponse)) * recip; //@?
			//};
		}

		virtual void AdjustTreeOutputs(RegressionTree& tree, DocumentPartitioning& partitioning, ScoreTracker& trainingScores) override
		{
			//AutoTimer timer("dynamic_pointer_cast<IStepSearch>(ObjectiveFunction))->AdjustTreeOutputs");
			for (int l = 0; l < tree.NumLeaves; l++)
			{
				Float output = 0.0;
				if (_bestStepRankingRegressionTrees)
				{ //@TODO 即使设置 这里 仍然和tlc不一致
					output = _learningRate * tree.GetOutput(l);
				}
				else
				{//@?    
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
				//PVAL3(tree.GetOutput(l), (_learningRate * (tree.GetOutput(l) + 1.4E-45)), (partitioning.Mean(_weights, Dataset.SampleWeights, l, false) + 1.4E-45));
				//PVAL2(l, output);
				//Pvector(_weights);
				tree.SetOutput(l, output);
			}
		}

	public:
		//virtual void GetGradientInOneQuery(int query, const Fvec& scores) override
		void GetGradientInOneQuery(int query, const Fvec& scores)
		{
			Float sigmoidParam = _learningRate;
			Float recip_npos = 1.0;
			Float recip_nneg = 1.0;
			if (_unbalancedSets)
			{
				recip_npos = 1.0 / ((Float)_npos);
				recip_nneg = 1.0 / ((Float)_nneg);
			}
			int label = Labels[query] ? 1 : -1;
			Float recip = Labels[query] ? recip_npos : recip_nneg;
			Float response = ((2.0 * label) * sigmoidParam) / (1.0 + std::exp(((2.0 * label) * sigmoidParam) * scores[query]));
			Float absResponse = std::abs(response);
			_gradient[query] = response * recip;
			_weights[query] = (absResponse * ((2.0 * sigmoidParam) - absResponse)) * recip; //@?
		}

	};

}  //----end of namespace gezi

#endif  //----end of BINARY_CLASSIFICATION_OBJECTIVE_FUNCTION_H_
