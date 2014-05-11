/**
 *  ==============================================================================
 *
 *          \file   Trainers/FastRank/Dataset.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-07 16:10:03.569728
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__FAST_RANK__DATASET_H_
#define TRAINERS__FAST_RANK__DATASET_H_
#include "Trainers/FastRank/Feature.h"
namespace gezi {

	struct FeatureBin
	{
		vector<Feature>& _features;
		int _doc;
		FeatureBin(vector<Feature>& features, int doc)
			:_features(features), _doc(doc)
		{

		}

		const double operator [] (int featureIdx) const
		{
			return _features[featureIdx].Bins[_doc];
		}
	};

	class Dataset
	{
	public:
		Dataset() = default;
		virtual ~Dataset() {}
		Dataset(Dataset&&) = default;
		Dataset& operator = (Dataset&&) = default;
		Dataset(const Dataset&) = default;
		Dataset& operator = (const Dataset&) = default;

		Dataset(vector<Feature>& features, vector<short>& ratings, dvec& weights)
			:Features(features), Ratings(ratings), Weights(weights)
		{
			for (auto feature : Features)
			{
				FeatureBinMedians.push_back(&feature.BinMedians);
			}
			NumDocs = features[0].Bins.Length();
			NumFeatures = Features.size();
		}

		FeatureBin GetFeatureBinRow(int index)
		{
			return FeatureBin(Features, index);
		}

		bool Empty()
		{
			return Features.empty();
		}
	public:
		int NumDocs = 0;
		int NumFeatures = 0;
		vector<Fvec*> FeatureBinMedians;
		vector<Feature> Features;
		vector<short> Ratings; //∂˛∑÷¿‡ 0,1 ≈≈–Ú ndcg 0,1,2,3,4 ?
		dvec Weights;
		dvec SampleWeights;
	protected:
	private:
		
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__DATASET_H_
