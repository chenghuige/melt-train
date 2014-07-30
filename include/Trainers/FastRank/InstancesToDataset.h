/**
 *  ==============================================================================
 *
 *          \file   Trainers/FastRank/InstancesToDataSet.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-07 16:09:54.350171
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__FAST_RANK__INSTANCES_TO_DATA_SET_H_
#define TRAINERS__FAST_RANK__INSTANCES_TO_DATA_SET_H_
#include "Prediction/Instances/Instances.h"
#include "Trainers/FastRank/Dataset.h"
#include "Numeric/BinFinder.h"
namespace gezi {

	//����InstancesҲ�����б���FeatureVectors 
	//1.���ո���Feature��ͳ����Ϣ��Ͱ ����ÿ��Feature��Ӧ��binUpperBounds, binMedians
	//2.��Ӧ����Feature,��������Instance�����Feature�϶�Ӧ��Ͱ��
	//TLC תfastrank���� ����zeroBin �����ϡ�� ���� �� zeroBin = 0 ��ʹ��ϡ���ʾ��Ͱ�ż�¼ ���￼��
	//ֻҪ��Feature��Ӧ��value 0Ϊ�� �Ƚ�ϡ�� ��ôͰ��¼�Ͳ���ϡ���ʾ 
	//@TODO fastrankʹ���ڲ�IntArray ����Ͱ����Ŀ ������Ҫ��bits ���� 2��Ͱ ֻ��Ҫ1��bit ����bool����
	//Ҳ����˵���ü���bits ���о����ܶ���ڴ�ռ������������Ϊ�˼� ��ʱ����int���� Feature::IntArray
	class InstancesToDataset
	{
	public:
		static IntArray GetBinValues(Vector& values, Fvec& upperBounds)
		{
			int zeroBin = first_ge(upperBounds, 0);
			IntArray bins(zeroBin, values.Length());
			values.ForEach([&](int idx, Float val) {
				bins.Add(idx, first_ge(upperBounds, val));
			});
			return bins;
		}

		static Dataset Convert(Instances& instances, int maxBins = 255, double sparsifyRatio = 0.3)
		{
			//-------------- ��ת��Ϊ��
			int numFeatures = instances.NumFeatures();
			vector<short> ratings;
			dvec weights;
			bool useWeight = false;
			vector<Vector> valuesVec(numFeatures, Vector(instances.size()));
			{
				//ProgressBar pb(instances.size(), "Converting from row format to column format");
				AutoTimer timer("Converting from row format to column format", 0);
				int numInstancesProcessed = 0;
				for (InstancePtr instance : instances)
				{
					//++pb;

					(instance->features).ForEach([&](int idx, Float val) {
						valuesVec[idx].Add(numInstancesProcessed, val);
					});

					ratings.push_back(instance->label);
					weights.push_back(instance->weight);

					if (instance->weight != 1.0)
					{//������Ǳ���2.0 ����Ĭ����Ȩ�ص� 
						useWeight = true;
					}

					numInstancesProcessed++;
				}
			}

			//------------- ��Ͱ ��ȡ bin upperbounds �� medians
			vector<Feature> features(numFeatures);
			{
				if (numFeatures > 10000)
				{
					ProgressBar pb(numFeatures, "Binning for bounds and medians");
					//AutoTimer timer("Binning for bounds and medians", 0);
					BinFinder binFinder;
#pragma omp parallel for firstprivate(pb) firstprivate(binFinder)
					for (int i = 0; i < numFeatures; i++)
					{
						++pb;
						Fvec values = valuesVec[i].Values(); //��һ��copy
						binFinder.FindBins(values, valuesVec[i].Length(), maxBins,
							features[i].BinUpperBounds, features[i].BinMedians);

						features[i].Name = instances.FeatureNames()[i];
					}
				}
				else
				{
					AutoTimer timer("Binning for bounds and medians", 0);
					BinFinder binFinder;
#pragma omp parallel for firstprivate(binFinder)
					for (int i = 0; i < numFeatures; i++)
					{
						Fvec values = valuesVec[i].Values(); //��һ��copy
						binFinder.FindBins(values, valuesVec[i].Length(), maxBins,
							features[i].BinUpperBounds, features[i].BinMedians);

						features[i].Name = instances.FeatureNames()[i];
					}
				}

			}

			//------------- �������instance��Ӧ����feature�ֵ���Ͱ�� bin 
			{
				//ProgressBar pb(numFeatures, "Get bin number for values");
				AutoTimer timer("Get bin number for values", 0);
				//#pragma omp parallel for firstprivate(pb) 
				for (int i = 0; i < numFeatures; i++)
				{
					//++pb;
					features[i].Bins = GetBinValues(valuesVec[i], features[i].BinUpperBounds);

					//PVAL3(i, features[i].Bins.Length(), features[i].BinUpperBounds.size());
					//PVECTOR(features[i].BinUpperBounds);

					features[i].Bins.Densify(sparsifyRatio);
					//features[i].Bins.ToDense();
				}
			}

			if (!useWeight)
			{
				weights.clear();
			}
			PVAL(useWeight);
			return Dataset(features, ratings, weights);
		}
	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__INSTANCES_TO_DATA_SET_H_
