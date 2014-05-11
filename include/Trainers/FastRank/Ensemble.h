/**
 *  ==============================================================================
 *
 *          \file   Ensemble.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 14:11:40.163962
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef ENSEMBLE_H_
#define ENSEMBLE_H_
#include "common_def.h"
#include "RegressionTree.h"
#include "Dataset.h"
namespace gezi {

	class Ensemble
	{
	private:
		double _bias = 0.0;
		string _firstInputInitializationContent;
		vector<RegressionTree> _trees;
	public:
		void AddTree(RegressionTree& tree) //@TODO RegressionTree&& ? python wrapper ok?
		{
			_trees.emplace_back(tree);
		}

		void AddTreeAt(RegressionTree& tree, int index)
		{
			_trees.emplace(_trees.begin() + index, tree);
		}

		double GetOutput(const FeatureBin& featureBins, int prefix)
		{
			double output = 0.0;
#pragma omp parallel for reduction(+: output)
			for (int h = 0; h < prefix; h++)
			{
				output += _trees[h].GetOutput(featureBins);
			}
			return output;
		}

		void GetOutputs(Dataset& dataset, dvec& outputs)
		{
			GetOutputs(dataset, outputs, -1);
		}

		void GetOutputs(Dataset& dataset, dvec& outputs, int prefix)
		{
			if ((prefix > _trees.size()) || (prefix < 0))
			{
				prefix = _trees.size();
			}
#pragma omp parallel for
			for (int doc = 0; doc < dataset.NumDocs; doc++)
			{
				outputs[doc] = GetOutput(dataset.GetFeatureBinRow(doc), prefix);
			}
		}

		void RemoveAfter(int index)
		{
			_trees.erase(_trees.begin() + index, _trees.end());
		}

		void RemoveTree(int index)
		{
			_trees.erase(_trees.begin() + index);
		}

		int NumTrees()
		{
			return _trees.size();
		}

		const RegressionTree& Tree() const
		{
			return _trees.back();
		}
	};

}  //----end of namespace gezi

#endif  //----end of ENSEMBLE_H_
