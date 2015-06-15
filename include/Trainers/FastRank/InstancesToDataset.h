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
#include "rabit_util.h"
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
			int zeroBin = gezi::first_ge(upperBounds, 0);
			IntArray bins(zeroBin, values.Length());
			values.ForEach([&](int idx, Float val) {
				bins.Add(idx, gezi::first_ge(upperBounds, val));
			});
			return bins;
		}

		//用于将Insatnces转换为trainset,首先计算分桶的BinUpperBounds等确定分桶的界限，然后所有特征值归一到桶序号
		//注意将会清空Instances
		static Dataset Convert(Instances& instances, int maxBins = 255, Float sparsifyRatio = 0.3)
		{
			//-------------- 行转换为列
			int numFeatures = instances.NumFeatures();
			int numDocs = instances.size();
			Fvec ratings, weights;
			bool useWeight = false;
			vector<Vector> valuesVec(numFeatures, Vector(numDocs));
			{
				//ProgressBar pb(instances.size(), "Converting from row format to column format");
				AutoTimer timer("Converting from row format to column format", 2);
				int numInstancesProcessed = 0;
				for (InstancePtr instance : instances)
				{
					//++pb;
					(instance->features).ForEach([&](int idx, Float val) {
						if (Rabit::Choose(idx))
							valuesVec[idx].Add(numInstancesProcessed, val);
					});

					ratings.push_back(instance->label);
					weights.push_back(instance->weight);

					if (instance->weight != 1.0)
					{//如果都是比如2.0 还是默认有权重的 
						useWeight = true;
					}
					numInstancesProcessed++;
					instance.reset();
				}
			}
			instances.clear();

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
						if (!Rabit::Choose(i))
							continue;
						++pb;
						Fvec values = valuesVec[i].Values(); //做一份copy
						binFinder.FindBins(values, valuesVec[i].Length(), maxBins,
							features[i].BinUpperBounds, features[i].BinMedians);

						//------------- 计算各个instance对应各个feature分到的桶号 bin 
						features[i].Bins = GetBinValues(valuesVec[i], features[i].BinUpperBounds);
						valuesVec[i].clear();
						features[i].Bins.Densify(sparsifyRatio);

						features[i].Name = instances.FeatureNames()[i];
					}
				}
				else
				{
					AutoTimer timer("Binning for bounds and medians", 2);
					BinFinder binFinder;
#pragma omp parallel for firstprivate(binFinder)
					for (int i = 0; i < numFeatures; i++)
					{
						if (!Rabit::Choose(i))
							continue;
						Fvec values = valuesVec[i].Values(); //做一份copy
						binFinder.FindBins(values, valuesVec[i].Length(), maxBins,
							features[i].BinUpperBounds, features[i].BinMedians);

						//------------- 计算各个instance对应各个feature分到的桶号 bin 
						features[i].Bins = GetBinValues(valuesVec[i], features[i].BinUpperBounds);
						valuesVec[i].clear();
						features[i].Bins.Densify(sparsifyRatio);

						features[i].Name = instances.FeatureNames()[i];
					}
				}

			}

			////------------- 计算各个instance对应各个feature分到的桶号 bin 
			//{
			//	//ProgressBar pb(numFeatures, "Get bin number for values");
			//	AutoTimer timer("Get bin number for values", 2);
			//	//#pragma omp parallel for firstprivate(pb) 
			//	for (int i = 0; i < numFeatures; i++)
			//	{
			//		if (!Rabit::Choose(i))
			//			continue;
			//		//++pb;
			//		features[i].Bins = GetBinValues(valuesVec[i], features[i].BinUpperBounds);
			//		valuesVec[i].clear();
			//		features[i].Bins.Densify(sparsifyRatio);
			//		//features[i].Bins.ToDense();
			//	}
			//}

			if (!useWeight)
			{
				weights.clear();
			}
			PVAL(useWeight);
			//如果是分布式按照feature切割 最后feature信息还要聚合 因为在RegressionTree::ToOnline还是需要全局信息的
			//@TODO 当前第一个分布式版本 只考虑并行加速 按feature分割 但是内存占用和单机版本是一样的
			//if (rabit::GetWorldSize() > 1)
			//{
			//	//@TODO Feature里面不都是基础类型
			//	//Rabit::Allreduce(features);
			//	gezi::Notifer notifer("Broadcast for features");
			//	Rabit::Broadcast(features);
			//}
			return Dataset(numDocs, features, ratings, weights);
		}

	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__INSTANCES_TO_DATA_SET_H_
