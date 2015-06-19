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

	//����ط��Ƿ���Կռ任ʱ�� ��������������� ����һ��ȫ����ά���� @TODO ��Feature����һ��DenseBins
	struct FeatureBin
	{
		const vector<Feature>& _features;
		int _doc;
		FeatureBin(const vector<Feature>& features, int doc)
			:_features(features), _doc(doc)
		{

		}

		Float operator [] (int featureIdx) const
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

		Dataset(int numDocs, vector<Feature>& features, vector<Float>& ratings, Fvec& weights)
			:NumDocs(numDocs), Features(move(features)), Ratings(move(ratings)), SampleWeights(move(weights))
		{
			for (auto feature : Features)
			{
				FeatureBinMedians.push_back(&feature.BinMedians);
			}
			NumFeatures = Features.size();
		}

		size_t size() const
		{
			return NumDocs;
		}

		//����row ����
		FeatureBin operator [] (int index)  const
		{
			return FeatureBin(Features, index);
		}

		FeatureBin GetFeatureBinRow(int index) const
		{
			return FeatureBin(Features, index);
		}

		bool Empty() const
		{
			return Features.empty();
		}

		svec FeatureNames()
		{
			return from(Features) 
				>> select([](const Feature& a) { return a.Name; }) 
				>> to_vector();
		}
	public:
		int NumDocs = 0;
		int NumFeatures = 0;
		vector<Fvec*> FeatureBinMedians;
		vector<Feature> Features;
		//vector<short> Ratings; //������ 0,1 ���� ndcg 0,1,2,3,4 ?
		Fvec Ratings;
		Fvec SampleWeights;
	protected:
	private:
		
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__DATASET_H_
