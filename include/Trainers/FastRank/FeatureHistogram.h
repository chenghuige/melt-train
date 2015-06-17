/**
 *  ==============================================================================
 *
 *          \file   FeatureHistogram.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 19:35:29.227506
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef FEATURE_HISTOGRAM_H_
#define FEATURE_HISTOGRAM_H_
#include "Feature.h"
#include "rabit_util.h"
namespace gezi {

	class FeatureHistogram
	{
	public:
		ivec CountByBin;
		bool IsSplittable = true;
		Float SumMedians = std::numeric_limits<Float>::quiet_NaN();
		Float SumMedianTargetProducts = std::numeric_limits<Float>::quiet_NaN();
		Float SumSquaredMedians = std::numeric_limits<Float>::quiet_NaN();
		Fvec SumTargetsByBin;
		Fvec SumWeightsByBin;
		int NumFeatureValues = 0;
		const Feature* Feature = NULL;

		FeatureHistogram() = default;
		FeatureHistogram(FeatureHistogram&&) = default;
		FeatureHistogram& operator = (FeatureHistogram&&) = default;
		FeatureHistogram(const FeatureHistogram&) = default;
		FeatureHistogram& operator = (const FeatureHistogram&) = default;

		FeatureHistogram(const gezi::Feature& feature, bool useWeights = false)
			:Feature(&feature), NumFeatureValues(feature.NumBins())
		{
			SumTargetsByBin.resize(NumFeatureValues);
			CountByBin.resize(NumFeatureValues);
			if (useWeights)
			{
				SumWeightsByBin.resize(NumFeatureValues);
			}
		}

		void Initialize(const gezi::Feature& feature, bool useWeights = false)
		{
			Feature = &feature;
			NumFeatureValues = feature.NumBins();
			SumTargetsByBin.resize(NumFeatureValues);
			CountByBin.resize(NumFeatureValues);
			if (useWeights)
			{
				SumWeightsByBin.resize(NumFeatureValues);
			}
		}

		//���д���@TODO �����õ� ע��swap�Ƿ�ok ������Ҫ��shared_ptr
		//@TODO ����ڲ�ֻ��swap moveֻ�ṩ int,Fvec&&,ivec&&�����Ľӿڸ���  ������ʾָ�� FeatureHistogram(255, move(sumTarget), move(binCount))������
		FeatureHistogram(int numBins, Fvec& sumTarget, ivec& binCount)
			:NumFeatureValues(numBins)
		{
			CHECK_EQ(sumTarget.size(), binCount.size()) << "length of input arrays is inconsistent";
			CHECK_LE(numBins, (int)sumTarget.size()) << "number of bins is greater than length of input array";
			SumTargetsByBin.swap(sumTarget);
			CountByBin.swap(binCount);
		}

		static int EstimateMemoryUsedForFeatureHistogram(int numBins, bool hasWeights)
		{
			return (((40 + (4 * numBins)) + (8 * numBins)) + (hasWeights ? (8 * numBins) : 0));
		}

		void ReassignFeature(const gezi::Feature& feature)
		{
			CHECK_LE(feature.NumBins(), (int)CountByBin.size()) << "new feature's bin number is larger than existing array size";
			Feature = &feature;
			NumFeatureValues = feature.NumBins();
		}

		void Subtract(FeatureHistogram& child)
		{
			CHECK_EQ(child.NumFeatureValues, NumFeatureValues) << "cannot subtract FeatureHistograms of different lengths";
			if (SumWeightsByBin.empty())
			{
				for (int i = 0; i < NumFeatureValues; i++)
				{
					SumTargetsByBin[i] -= child.SumTargetsByBin[i];
					CountByBin[i] -= child.CountByBin[i];
				}
			}
			else
			{
				for (int i = 0; i < NumFeatureValues; i++)
				{
					SumTargetsByBin[i] -= child.SumTargetsByBin[i];
					CountByBin[i] -= child.CountByBin[i];
					SumWeightsByBin[i] -= child.SumWeightsByBin[i];
				}
			}
		}

		struct SumupInputData
		{
			const Fvec& BinMedians;
			const ivec& DocIndices;
			const Fvec& Outputs;
			const Fvec& Weights;
			Float SumTargets;
			Float SumWeights;
			int TotalCount;
			SumupInputData(int totalCount, Float sumTargets, Float sumWeights, const Fvec& outputs,
				const Fvec& weights, const ivec& docIndices, const Fvec& binMedians)
				:TotalCount(totalCount), SumTargets(sumTargets), SumWeights(sumWeights), Outputs(outputs), 
				Weights(weights), DocIndices(docIndices), BinMedians(binMedians)
			{
			}
		};

		inline void Sumup(int featureIndex, int numDocsInLeaf, Float sumTargets,
			Fvec& outputs, ivec& docIndices)
		{
			Fvec weights;
			SumupWeighted(featureIndex, numDocsInLeaf, sumTargets, 0.0, outputs, weights, docIndices);
		}

		//������ڵ�ǰfeatureIndexͳ��������Ӧbins��ÿ��bin���ۼ�targetֵ Ҳ���Ǽ���ֱ��ͼ��
		inline void SumupWeighted(int featureIndex, int numDocsInLeaf, Float sumTargets, Float sumWeights,
			const Fvec& outputs, const Fvec& weights, const ivec& docIndices)
		{
			SumupInputData input(numDocsInLeaf, sumTargets, sumWeights, outputs, weights, docIndices, Feature->BinMedians);
			gezi::zeroset(SumTargetsByBin);
			if (!SumWeightsByBin.empty())
			{
				gezi::zeroset(SumWeightsByBin);
			}
			gezi::zeroset(CountByBin);
			SumMedians = SumSquaredMedians = SumMedianTargetProducts = std::numeric_limits<Float>::quiet_NaN();
			Sumup(Feature->Bins, input);
		}

	protected:
		inline void Sumup(const IntArray& bins, const SumupInputData& input)
		{
			if (input.DocIndices.empty())
			{
				if (SumWeightsByBin.empty())
				{
					SumupRoot(bins, input);
				}
				else
				{
					SumupRootWeighted(bins, input);
				}
			}
			else
			{
				if (SumWeightsByBin.empty())
				{
					SumupLeaf(bins, input);
				}
				else
				{
					SumupLeafWeighted(bins, input);
				}
			}
		}

		inline void SumupRoot(const IntArray& bins, const SumupInputData& input)
		{
			if (bins.IsDense())
			{
				SumupRootDense(bins, input);
			}
			else
			{
				SumupRootSparse(bins, input);
			}
		}

		inline void SumupRootDense(const IntArray& bins, const SumupInputData& input)
		{
			bins.ForEachDense([&, this](int index, int featureBin)
			{
				Float output = input.Outputs[index];
				SumTargetsByBin[featureBin] += output;
				CountByBin[featureBin]++;
			});
		}

		inline void SumupRootSparse(const IntArray& bins, const SumupInputData& input)
		{
			Float totalOutput = 0.0;
			bins.ForEachSparse([&, this](int index, int featureBin)
			{
				Float output = input.Outputs[index];
				SumTargetsByBin[featureBin] += output;
				CountByBin[featureBin]++;
				totalOutput += output;
			});

			SumTargetsByBin[bins.ZeroValue()] += input.SumTargets - totalOutput;
			CountByBin[bins.ZeroValue()] += input.TotalCount - bins.Count();
		}

		void SumupRootWeighted(const IntArray& bins, const SumupInputData& input)
		{
			if (bins.IsDense())
			{
				bins.ForEachDense([&, this](int index, int featureBin)
				{
					Float output = input.Outputs[index];
					SumTargetsByBin[featureBin] += output;
					Float weight = input.Weights[index];
					SumWeightsByBin[featureBin] += weight;
					CountByBin[featureBin]++;
				});
			}
			else
			{
				Float totalOutput = 0.0;
				Float totalWeight = 0.0;
				bins.ForEachSparse([&, this](int index, int featureBin)
				{
					Float output = input.Outputs[index];
					SumTargetsByBin[featureBin] += output;
					Float weight = input.Weights[index];
					SumWeightsByBin[featureBin] += weight;
					CountByBin[featureBin]++;
					totalOutput += output;
					totalWeight += weight;
				});

				SumTargetsByBin[bins.ZeroValue()] += input.SumTargets - totalOutput;
				SumWeightsByBin[bins.ZeroValue()] += input.SumWeights - totalWeight;
				CountByBin[bins.ZeroValue()] += input.TotalCount - bins.Count();
			}
		}

		inline void SumupLeaf(const IntArray& bins, const SumupInputData& input)
		{
			//VLOG(2) << "SumupLeaf";
			if (bins.IsDense())
			{
				SumupLeafDense(bins, input);
			}
			else
			{
				SumupLeafSparse(bins, input);
			}
		}

		//@TODO ò�ƻ��Ǳ�tlc��һЩ ����Ϊtlc��IntArray��ͬ������bit4 bit8?
		inline void SumupLeafDense(const IntArray& bins, const SumupInputData& input)
		{
			for (int iDocIndices = 0; iDocIndices < input.TotalCount; iDocIndices++)
			{
				Float output = input.Outputs[iDocIndices];
				int index = input.DocIndices[iDocIndices];
				int featureBin = bins.values[index];
				SumTargetsByBin[featureBin] += output;
				CountByBin[featureBin]++;
			}
		}

		inline void SumupLeafSparse(const IntArray& bins, const SumupInputData& input)
		{
			int iDocIndices = 0;
			int totalCount = 0;
			Float totalOutput = 0.0;

			int len = bins.indices.size();
			for (int i = 0; i < len; i++)
			{
				int index = bins.indices[i];
				while (index > input.DocIndices[iDocIndices])
				{
					iDocIndices++;
					if (iDocIndices >= input.TotalCount)
						goto end;
				}
				if (index == input.DocIndices[iDocIndices])
				{
					Float output = input.Outputs[iDocIndices];
					int featureBin = bins.values[i];
					SumTargetsByBin[featureBin] += output;
					totalOutput += output;
					CountByBin[featureBin]++;
					totalCount++;
					iDocIndices++;
					if (iDocIndices >= input.TotalCount)
						break;
				}
			}
		end:
			SumTargetsByBin[bins.ZeroValue()] += input.SumTargets - totalOutput;
			CountByBin[bins.ZeroValue()] += input.TotalCount - totalCount;
		}

		void SumupLeafWeighted(const IntArray& bins, const SumupInputData& input)
		{
			if (bins.IsDense())
			{
				for (int iDocIndices = 0; iDocIndices < input.TotalCount; iDocIndices++)
				{
					Float output = input.Outputs[iDocIndices];
					Float weight = input.Weights[iDocIndices];
					int index = input.DocIndices[iDocIndices];
					int featureBin = bins.values[index];
					SumTargetsByBin[featureBin] += output;
					SumWeightsByBin[featureBin] += weight;
					CountByBin[featureBin]++;
				}
			}
			else
			{
				int iDocIndices = 0;
				int totalCount = 0;
				Float totalOutput = 0.0;
				Float totalWeight = 0.0;

				int len = bins.indices.size();
				for (int i = 0; i < len; i++)
				{
					int index = bins.indices[i];
					while (index > input.DocIndices[iDocIndices])
					{
						iDocIndices++;
						if (iDocIndices >= input.TotalCount)
							goto end;
					}
					if (index == input.DocIndices[iDocIndices])
					{
						Float output = input.Outputs[iDocIndices];
						Float weight = input.Weights[iDocIndices];
						int featureBin = bins.values[i];
						SumTargetsByBin[featureBin] += output;
						SumWeightsByBin[featureBin] += weight;
						totalOutput += output; 
						totalWeight += weight;
						CountByBin[featureBin]++;
						totalCount++;
						iDocIndices++;
						if (iDocIndices >= input.TotalCount)
							break;
					}
				}
			end:
				SumTargetsByBin[bins.ZeroValue()] += input.SumTargets - totalOutput;
				SumWeightsByBin[bins.ZeroValue()] += input.SumWeights - totalWeight;
				CountByBin[bins.ZeroValue()] += input.TotalCount - totalCount;
			}
		}
	};

}  //----end of namespace gezi

#endif  //----end of FEATURE_HISTOGRAM_H_
