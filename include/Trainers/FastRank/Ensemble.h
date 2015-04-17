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
#include "common_util.h"
#include "RegressionTree.h"
#include "Dataset.h"
namespace gezi {

	class Ensemble
	{
	private:
		Float _bias = 0.0;
		string _firstInputInitializationContent;
		vector<RegressionTree> _trees;
	public:
		vector<OnlineRegressionTree> ToOnline(vector<Feature>& features)
		{
			vector<OnlineRegressionTree> trees;
			for (auto& tree : _trees)
			{
				tree.ToOnline(features);
				trees.emplace_back((OnlineRegressionTree)tree);
			}
			return trees;
		}
		void AddTree(RegressionTree& tree) //@TODO RegressionTree&& ? python wrapper ok?
		{
			_trees.emplace_back(tree);
		}

		void AddTreeAt(RegressionTree& tree, int index)
		{
			_trees.emplace(_trees.begin() + index, tree);
		}

		Float GetOutput(const FeatureBin& featureBins, int prefix)
		{
			Float output = 0.0;
#pragma omp parallel for reduction(+: output)
			for (int h = 0; h < prefix; h++)
			{
				output += _trees[h].GetOutput(featureBins);
			}
			return output;
		}

		void GetOutputs(Dataset& dataset, Fvec& outputs)
		{
			GetOutputs(dataset, outputs, -1);
		}

		void GetOutputs(Dataset& dataset, Fvec& outputs, int prefix)
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

		RegressionTree& Tree()
		{
			return _trees.back();
		}

		RegressionTree& LastTree()
		{
			return _trees.back();
		}

		RegressionTree& Back()
		{
			return _trees.back();
		}

		//统计前prefix棵数目一般用所有的树
		map<int, Float> GainMap(int prefix, bool normalize)
		{
			map<int, Float> m;
			if (_trees.empty())
			{
				return m;
			}
			if ((prefix > NumTrees()) || (prefix < 0))
			{
				prefix = NumTrees();
			}
			for (int i = 0; i < prefix; i++)
			{
				_trees[i].GainMap(m);
			}

			if (normalize)
			{
				for (auto item : m)
				{
					item.second /= (Float)NumTrees();
				}
			}

			return m;
		}


		string ToGainSummary(vector<Feature>& featureList, int prefix = -1, bool includeZeroGainFeatures = true, bool normalize = true)
		{
			map<int, Float> m = GainMap(prefix, normalize);
			if (includeZeroGainFeatures)
			{
				for (size_t k = 0; k < featureList.size(); k++)
				{
					add_value(m, k, 0.0);
				}
			}

			// @TODO
			//vector<pair<int, Float> > sortedByGain = sort_map_by_value_reverse(m);
			//auto sortedByGain = sort_map_by_value_reverse(m);
			vector<pair<int, Float> > sortedByGain;
			sort_map_by_value_reverse(m, sortedByGain);
			Float maxValue = sortedByGain[0].second;
			Float normalizingFactor = (normalize && (maxValue != 0.0)) ? std::sqrt(maxValue) : 1.0;
			Float power = normalize ? 0.5 : 1.0;

			stringstream ss;
			int id = 0;
			for (auto item : sortedByGain)
			{
				/*	ss << "f_" << STR(id++) << ":"  << item.first << ":"
						<< featureList[item.first].Name
						<< setiosflags(ios::left) << setfill(' ') << setw(40)
						<< " " << std::pow(item.second, power) / normalizingFactor << endl;*/
				ss << setiosflags(ios::left) << setfill(' ') << setw(40)
					<< "f_" + STR(id++) + ":" + STR(item.first) + ":" + featureList[item.first].Name
					<< " " << std::pow(item.second, power) / normalizingFactor << endl;
			}
			return ss.str();
		}

	};

}  //----end of namespace gezi

#endif  //----end of ENSEMBLE_H_
