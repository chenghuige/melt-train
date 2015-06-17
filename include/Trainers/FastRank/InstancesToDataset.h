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
			int zeroBin = gezi::first_ge(upperBounds, 0);
			IntArray bins(zeroBin, values.Length());
			values.ForEach([&](int idx, Float val) {
				bins.Add(idx, gezi::first_ge(upperBounds, val));
			});
			return bins;
		}

		//���ڽ�Insatncesת��Ϊtrainset,���ȼ����Ͱ��BinUpperBounds��ȷ����Ͱ�Ľ��ޣ�Ȼ����������ֵ��һ��Ͱ���
		//ע�⽫�����Instances
		static Dataset Convert(Instances& instances, int maxBins = 255, Float sparsifyRatio = 0.3)
		{
			//-------------- ��ת��Ϊ��
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
					{//������Ǳ���2.0 ����Ĭ����Ȩ�ص� 
						useWeight = true;
					}
					numInstancesProcessed++;
					instance.reset();
				}
			}
			instances.clear();

			//------------- ��Ͱ ��ȡ bin upperbounds �� medians
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

					Fvec values = valuesVec[i].Values(); //��һ��copy
					int length = valuesVec[i].Length();

					if (Rabit::GetWorldSize() > 1 && FLAGS_distributeMode == 2)
					{//����instance���зָ�, bin normalizing����͵�������һ��, Ҳ������һ���Ǿ�ȷ�ķֲ�ʽ�汾 ʹ��һ�����ݼ���bin����Ϣ Ȼ��broadcast�������ڵ㹲ͬʹ��
						//����allreduce�������BinUpperBoundsȡ���ֵ,���ܻ���ϸ΢�Ĳ�һ�� @TODO ���Ǽ�����С�ܶ�
#pragma omp ordered
						{
							gezi::Notifer notifer("Allgather values", 2);
							Rabit::Allgather(values);
							Rabit::Allreduce<rabit::op::Sum>(length);
						}
					}

					//@TODO �����Ӧinstances�зֵ�ʹ��allgather����� ����Ŀǰrabit��֧��
					binFinder.FindBins(values, length, maxBins,
						features[i].BinUpperBounds, features[i].BinMedians);

					if (Rabit::GetWorldSize() > 1 && FLAGS_distributeMode > 2)
					{ //�Ǿ�ȷģʽ ʹ��root�Ľ��
#pragma omp ordered
						{
							gezi::Notifer notifer("Broadcast bounds", 2);
							Rabit::Broadcast(features[i].BinUpperBounds, 0);
							Rabit::Broadcast(features[i].BinMedians, 0);
						}
					}

					//------------- �������instance��Ӧ����feature�ֵ���Ͱ�� bin 
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

			//����Ƿֲ�ʽ����feature�и� ���feature��Ϣ��Ҫ�ۺ� ��Ϊ��RegressionTree::ToOnline������Ҫȫ����Ϣ��
			//distributeMode = 0 ֻ���ǲ��м��� ��feature�ָ� �����ڴ�ռ�ú͵����汾��һ���� distributeMode = 1��ʱ�������ϻ���ռ�ÿռ�(@TODO check),���ǻ����Ӽ��㽻������
			if (Rabit::GetWorldSize() > 1 && FLAGS_distributeMode == 0)
			{
				gezi::Notifer notifer("BroadcastAsString features", 1);
				//@TODO Feature���治���ǻ�������
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
