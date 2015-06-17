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
DECLARE_int32(distributeMode);
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
			PVAL(numDocs);
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
						if (FLAGS_distributeMode > 1 || Rabit::Choose(idx))
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
				ProgressBar pb(numFeatures, "Instance2Datset-Binning");
				//AutoTimer timer("Binning for bounds and medians", 0);
				BinFinder binFinder;
#pragma omp parallel for ordered firstprivate(pb) firstprivate(binFinder)
				for (int i = 0; i < numFeatures; i++)
				{
					++pb;

					if (FLAGS_distributeMode < 2 && !Rabit::Choose(i))
						continue;

					Fvec values = valuesVec[i].Values(); //做一份copy
					int length = valuesVec[i].Length();

					if (Rabit::GetWorldSize() > 1 && FLAGS_distributeMode == 2)
					{//按照instance进行分割, bin normalizing结果和单机保持一致, 也可以有一个非精确的分布式版本 使用一份数据计算bin的信息 然后broadcast到各个节点共同使用
						//或者allreduce处理各个BinUpperBounds取最大值,可能会有细微的不一致 @TODO 但是计算量小很多
#pragma omp ordered
						{
							gezi::Notifer notifer("Allgather values", 2);
							Rabit::Allgather(values);
							Rabit::Allreduce<rabit::op::Sum>(length);
						}
					}

					//@TODO 这里对应instances切分的使用allgather最合适 但是目前rabit不支持
					binFinder.FindBins(values, length, maxBins,
						features[i].BinUpperBounds, features[i].BinMedians);

					if (Rabit::GetWorldSize() > 1 && FLAGS_distributeMode > 2)
					{ //非精确模式 使用root的结果
#pragma omp ordered
						{
							gezi::Notifer notifer("Broadcast bounds", 2);
							Rabit::Broadcast(features[i].BinUpperBounds, 0);
							Rabit::Broadcast(features[i].BinMedians, 0);
						}
					}

					//------------- 计算各个instance对应各个feature分到的桶号 bin 
					features[i].Bins = GetBinValues(valuesVec[i], features[i].BinUpperBounds);
					valuesVec[i].clear();
					features[i].Bins.Densify(sparsifyRatio);

					features[i].Name = instances.FeatureNames()[i];
				}
			}

			if (!useWeight)
			{
				weights.clear();
			}
			PVAL(useWeight);

			//如果是分布式按照feature切割 最后feature信息还要聚合 因为在RegressionTree::ToOnline还是需要全局信息的
			//distributeMode = 0 只考虑并行加速 按feature分割 但是内存占用和单机版本是一样的 distributeMode = 1的时候理论上会少占用空间(@TODO check),但是会增加计算交互代价
			if (Rabit::GetWorldSize() > 1 && FLAGS_distributeMode == 0)
			{
				gezi::Notifer notifer("BroadcastAsString features", 1);
				//@TODO Feature里面不都是基础类型
				//Rabit::Allreduce(features);
				Rabit::BroadcastAsString(features);
			}

			return Dataset(numDocs, features, ratings, weights);
		}

	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__INSTANCES_TO_DATA_SET_H_
