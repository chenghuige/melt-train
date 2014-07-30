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

	//输入Instances也就是行表达的FeatureVectors 
	//1.按照各个Feature列统计信息分桶 产出每个Feature对应的binUpperBounds, binMedians
	//2.对应各个Feature,计算所有Instance在这个Feature上对应的桶号
	//TLC 转fastrank策略 计算zeroBin 如果是稀疏 并且 是 zeroBin = 0 才使用稀疏表示的桶号记录 这里考虑
	//只要是Feature对应的value 0为主 比较稀疏 那么桶记录就采用稀疏表示 
	//@TODO fastrank使用内部IntArray 对于桶的数目 计算需要的bits 比如 2个桶 只需要1个bit 类似bool类型
	//也就是说利用计算bits 进行尽可能多的内存占用缩减，这里为了简单 暂时按照int处理 Feature::IntArray
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
			//-------------- 行转换为列
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
					{//如果都是比如2.0 还是默认有权重的 
						useWeight = true;
					}

					numInstancesProcessed++;
				}
			}

			//------------- 分桶 获取 bin upperbounds 和 medians
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
						Fvec values = valuesVec[i].Values(); //做一份copy
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
						Fvec values = valuesVec[i].Values(); //做一份copy
						binFinder.FindBins(values, valuesVec[i].Length(), maxBins,
							features[i].BinUpperBounds, features[i].BinMedians);

						features[i].Name = instances.FeatureNames()[i];
					}
				}

			}

			//------------- 计算各个instance对应各个feature分到的桶号 bin 
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
