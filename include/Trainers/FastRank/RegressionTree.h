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
#include "Feature.h"
namespace gezi {

	class RegressionTree : public OnlineRegressionTree
	{
	public:
		RegressionTree(int maxLeaves)
		{
			_weight = 1.0;
			Reset(maxLeaves);
			NumLeaves = 1;
		}

		void Reset(int maxLeaves)
		{
			_splitFeature.resize(maxLeaves - 1);
			_splitGain.resize(maxLeaves - 1);
			_gainPValue.resize(maxLeaves - 1);
			_previousLeafValue.resize(maxLeaves - 1);
			_threshold.resize(maxLeaves - 1); //online是double离线训练其实是uint 
			_lteChild.resize(maxLeaves - 1);
			_gtChild.resize(maxLeaves - 1);
			_leafValue.resize(maxLeaves);
			_parent.resize(maxLeaves);
		}

		void Print()
		{
			Pvector(_splitFeature);
			Pvector(_splitGain);
			Pvector(_gainPValue);
			Pvector(_lteChild);
			Pvector(_gtChild);
			Pvector(_threshold);
			Pvector(_leafValue);
			Pval(NumLeaves);
		}

		void Print(vector<Feature>& features)
		{
			//Pvector(_splitFeature);
			Pvector(_splitGain);
			/*		Pvector(_gainPValue);
					Pvector(_lteChild);
					Pvector(_gtChild);
					Pval(NumLeaves);
					dvec threshold;
					for (size_t i = 0; i < _threshold.size(); i++)
					{
					uint val = (uint)_threshold[i];
					threshold.push_back(features[_splitFeature[i]].BinUpperBounds[val]);
					}
					Pvector(_threshold);
					Pvector(threshold)*/;
			Pvector(_leafValue);
		}

		void Finalize()
		{
			Reset(NumLeaves);
		}

		void ToOnline(vector<Feature>& features)
		{
			for (size_t i = 0; i < _threshold.size(); i++)
			{
				uint val = (uint)_threshold[i];
				_threshold[i] = features[_splitFeature[i]].BinUpperBounds[val];
			}
		}

		//score已经resize好
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
			return ~node; //~ means -node - 1 (~-3) --- [2]
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
			int indexOfNewInternal = NumLeaves - 1;
			/*	int parent = find_index(_lteChild, ~leaf);
				if (parent < _lteChild.size())
				{
				_lteChild[parent] = indexOfNewInternal;
				}
				else
				{
				parent = find_index(_gtChild, ~leaf);
				if (parent < _gtChild.size())
				{
				_gtChild[parent] = indexOfNewInternal;
				}
				}*/
			if (NumLeaves > 1)
			{
				int parent = _parent[leaf];
				if (parent >= 0)
				{
					_lteChild[parent] = indexOfNewInternal;
				}
				else
				{
					_gtChild[~parent] = indexOfNewInternal;
				}
			}
			_splitFeature[indexOfNewInternal] = feature;
			_splitGain[indexOfNewInternal] = gain;
			_gainPValue[indexOfNewInternal] = gainPValue;
			_threshold[indexOfNewInternal] = threshold;
			_lteChild[indexOfNewInternal] = ~leaf;
			_previousLeafValue[indexOfNewInternal] = _leafValue[leaf];
			_leafValue[leaf] = LTEValue;
			_parent[leaf] = indexOfNewInternal;
			_gtChild[indexOfNewInternal] = ~NumLeaves;
			_leafValue[NumLeaves] = GTValue;
			_parent[NumLeaves] = ~indexOfNewInternal;
			if (LTEValue > _maxOutput)
			{
				_maxOutput = LTEValue;
			}
			if (GTValue > _maxOutput)
			{
				_maxOutput = GTValue;
			}
			NumLeaves++;
			return indexOfNewInternal;
		}

		int SplitFeature(int node)
		{
			return _splitFeature[node];
		}

		void SetOutput(int leaf, double value)
		{
			_leafValue[leaf] = value;
		}

		map<int, double> GainMap()
		{
			map<int, double> m;
			GainMap(m);
			return m;
		}

		void GainMap(map<int, double>& m)
		{
			int numInternals = NumLeaves - 1;
			for (int n = 0; n < numInternals; n++)
			{
				add_value(m, _splitFeature[n], _splitGain[n]);
			}
		}

		const int NumNodes() const
		{
			return NumLeaves - 1;
		}

	private:
		ivec _parent;
	};

}  //----end of namespace gezi

#endif  //----end of REGRESSION_TREE_H_
