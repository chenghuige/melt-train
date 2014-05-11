/**
 *  ==============================================================================
 *
 *          \file   RegressionTree.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-08 19:22:50.386968
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef REGRESSION_TREE_H_
#define REGRESSION_TREE_H_

#include "Trainers/FastRank/OnlineRegressionTree.h"
#include "Dataset.h"
namespace gezi {

	class RegressionTree : public OnlineRegressionTree
	{
	public:
		RegressionTree(int maxLeaves)
		{
			_weight = 1.0;
			_splitFeature.resize(maxLeaves - 1);
			_splitGain.resize(maxLeaves - 1);
			_gainPValue.resize(maxLeaves - 1);
			_previousLeafValue.resize(maxLeaves - 1);
			_threshold.resize(maxLeaves - 1);
			_lteChild.resize(maxLeaves - 1);
			_gtChild.resize(maxLeaves - 1);
			_leafValue.resize(maxLeaves);
			NumLeaves = 1;
		}

		//score“—æ≠resize∫√
		void AddOutputsToScores(Dataset& dataset, dvec& scores)
		{
#pragma omp parallel for
			for (int d = 0; d < dataset.NumDocs; d++)
			{
				scores[d] += GetOutput(dataset.GetFeatureBinRow(d));
			}
		}

		void AddOutputsToScores(Dataset& dataset, dvec& scores, double multiplier)
		{
			for (int d = 0; d < dataset.NumDocs; d++)
			{
				scores[d] += multiplier * GetOutput(dataset.GetFeatureBinRow(d));
			}
		}

		double GetOutput(const FeatureBin& featureBin)
		{
			if (_lteChild[0] == 0)
			{
				return 0.0;
			}
			int leaf = GetLeaf(featureBin);
			return GetOutput(leaf);
		}

		int GetLeaf(const FeatureBin& featureBin)
		{
			if (NumLeaves == 1)
			{
				return 0;
			}
			int node = 0;

			while (node >= 0)
			{
				if (featureBin[_splitFeature[node]] <= _threshold[node])
				{
					node = _lteChild[node];
				}
				else
				{
					node = _gtChild[node];
				}
			}
			return ~node;
		}

		//@TODO range ? for IEnumerable ?
		ivec GetNodesLeaves(int node)
		{
			if (NumLeaves == 1)
			{
				return ivec(1, 0);
			}
			if (node < 0)
			{
				return ivec(~node, 1);
			}

			//@TODO try cpplinq concatenate
			ivec left = GetNodesLeaves(_lteChild[node]);
			ivec right = GetNodesLeaves(_gtChild[node]);
			left.insert(left.end(), right.begin(), right.end());
			return left;
		}

		double GetOutput(int leaf)
		{
			return _leafValue[leaf];
		}

		dvec GetOutputs(Dataset& dataset)
		{
			dvec outputs(dataset.NumDocs);
			for (int d = 0; d < dataset.NumDocs; d++)
			{
				outputs[d] = GetOutput(dataset.GetFeatureBinRow(d));
			}
			return outputs;
		}

		int GTChild(int node)
		{
			return _gtChild[node];
		}

		double LeafValue(int leaf)
		{
			return _leafValue[leaf];
		}

		int LTEChild(int node)
		{
			return _lteChild[node];
		}

		int Split(int leaf, int feature, uint threshold, double LTEValue, double GTValue, double gain, double gainPValue)
		{
			int indexOfNewNonLeaf = NumLeaves - 1;
			int parent = find_index(_lteChild, ~leaf);
			if (parent < _lteChild.size())
			{
				_lteChild[parent] = indexOfNewNonLeaf;
			}
			else
			{
				parent = find_index(_gtChild, ~leaf);
				if (parent < _gtChild.size())
				{
					_gtChild[parent] = indexOfNewNonLeaf;
				}
			}
			_splitFeature[indexOfNewNonLeaf] = feature;
			_splitGain[indexOfNewNonLeaf] = gain;
			_gainPValue[indexOfNewNonLeaf] = gainPValue;
			_threshold[indexOfNewNonLeaf] = threshold;
			_lteChild[indexOfNewNonLeaf] = ~leaf;
			_previousLeafValue[indexOfNewNonLeaf] = _leafValue[leaf];
			_leafValue[leaf] = LTEValue;
			_gtChild[indexOfNewNonLeaf] = ~NumLeaves;
			_leafValue[NumLeaves] = GTValue;
			if (LTEValue > _maxOutput)
			{
				_maxOutput = LTEValue;
			}
			if (GTValue > _maxOutput)
			{
				_maxOutput = GTValue;
			}
			NumLeaves++;
			return indexOfNewNonLeaf;
		}

		int SplitFeature(int node)
		{
			return _splitFeature[node];
		}

		void SetOutput(int leaf, double value)
		{
			//if (this is AffineRegressionTree)
			//{
			//	throw new InvalidOperationException("Cannot set output of affine leaf trees.");
			//}
			_leafValue[leaf] = value;
		}

		map<int, double> GainMap(bool normalize = true)
		{
			map<int, double> m;
			GainMap(m);
			return m;
		}

		void GainMap(map<int, double>& m, bool normalize = true)
		{
			int numNonLeaves = NumLeaves - 1;
			for (int n = 0; n < numNonLeaves; n++)
			{
				add_value(m, _splitFeature[n], _splitGain[n]);
			}
		}
	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of REGRESSION_TREE_H_
