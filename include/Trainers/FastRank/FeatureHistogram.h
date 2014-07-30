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
namespace gezi {

	class FeatureHistogram
	{
	public:
		ivec CountByBin;
		bool IsSplittable = true;
		double SumMedians = std::numeric_limits<double>::quiet_NaN();
		double SumMedianTargetProducts = std::numeric_limits<double>::quiet_NaN();
		double SumSquaredMedians = std::numeric_limits<double>::quiet_NaN();
		dvec SumTargetsByBin;
		dvec SumWeightsByBin;
		int NumFeatureValues = 0;
		Feature* Feature = NULL;

		FeatureHistogram() = default;
		FeatureHistogram(FeatureHistogram&&) = default;
		FeatureHistogram& operator = (FeatureHistogram&&) = default;
		FeatureHistogram(const FeatureHistogram&) = default;
		FeatureHistogram& operator = (const FeatureHistogram&) = default;

		FeatureHistogram(gezi::Feature& feature, bool useWeights = false)
			:Feature(&feature), NumFeatureValues(feature.NumBins())
		{
			SumTargetsByBin.resize(NumFeatureValues);
			CountByBin.resize(NumFeatureValues);
			if (useWeights)
			{
				SumWeightsByBin.resize(NumFeatureValues);
			}
		}

		void Initialize(gezi::Feature& feature, bool useWeights = false)
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

		//并行处理@TODO 可能用到 注意swap是否ok 还是需要用shared_ptr
		FeatureHistogram(int numBins, dvec& sumTarget, ivec& binCount)
			:NumFeatureValues(numBins)
		{
			if (sumTarget.size() != binCount.size())
			{
				THROW("length of input arrays is inconsistent");
			}
			if (numBins > (int)sumTarget.size())
			{
				THROW("number of bins is greater than length of input array");
			}
			SumTargetsByBin.swap(sumTarget);
			CountByBin.swap(binCount);
		}

		static int EstimateMemoryUsedForFeatureHistogram(int numBins, bool hasWeights)
		{
			return (((40 + (4 * numBins)) + (8 * numBins)) + (hasWeights ? (8 * numBins) : 0));
		}

		void ReassignFeature(gezi::Feature& feature)
		{
			if (feature.NumBins() > CountByBin.size())
			{
				THROW("new feature's bin number is larger than existing array size");
			}
			Feature = &feature;
			NumFeatureValues = feature.NumBins();
		}

		void Subtract(FeatureHistogram& child)
		{
			if (child.NumFeatureValues != NumFeatureValues)
			{
				THROW("cannot subtract FeatureHistograms of different lengths");
			}
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
			dvec& BinMedians;
			ivec& DocIndices;
			dvec& Outputs;
			dvec& Weights;
			double SumTargets;
			double SumWeights;
			int TotalCount;
			SumupInputData(int totalCount, double sumTargets, double sumWeights, dvec& outputs,
				dvec& weights, ivec& docIndices, dvec& binMedians)
				:TotalCount(totalCount), SumTargets(sumTargets), SumWeights(sumWeights), Outputs(outputs), Weights(weights), DocIndices(docIndices), BinMedians(binMedians)
			{
			}
		};

		inline void Sumup(int featureIndex, int numDocsInLeaf, double sumTargets,
			dvec& outputs, ivec& docIndices)
		{
			dvec weights;
			SumupWeighted(featureIndex, numDocsInLeaf, sumTargets, 0.0, outputs, weights, docIndices);
		}

		//这里对于当前featureIndex统计了它对应bins的每个bin的累加target值 也就是计算直方图了
		inline void SumupWeighted(int featureIndex, int numDocsInLeaf, double sumTargets, double sumWeights,
			dvec& outputs, dvec& weights, ivec& docIndices)
		{
			SumupInputData input(numDocsInLeaf, sumTargets, sumWeights, outputs, weights, docIndices, Feature->BinMedians);
			zeroset(SumTargetsByBin);
			if (!SumWeightsByBin.empty())
			{
				zeroset(SumWeightsByBin);
			}
			zeroset(CountByBin);
			SumMedians = SumSquaredMedians = SumMedianTargetProducts = std::numeric_limits<double>::quiet_NaN();
			Sumup(Feature->Bins, input);
		}

	protected:
		inline void Sumup(IntArray& bins, SumupInputData& input)
		{
			//AutoTimer timer("Sumup");
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

		inline void SumupRoot(IntArray& bins, SumupInputData& input)
		{
			//VLOG(2) << "SumupRoot";
			if (bins.IsDense())
			{
				SumupRootDense(bins, input);
			}
			else
			{
				SumupRootSparse(bins, input);
			}
		}

		inline void SumupRootDense(IntArray& bins, SumupInputData& input)
		{
			bins.ForEachDense([&, this](int index, int featureBin)
			{
				double output = input.Outputs[index];
				SumTargetsByBin[featureBin] += output;
				CountByBin[featureBin]++;
			});
		}

		inline void SumupRootSparse(IntArray& bins, SumupInputData& input)
		{
			double totalOutput = 0.0;
			bins.ForEachSparse([&, this](int index, int featureBin)
			{
				double output = input.Outputs[index];
				SumTargetsByBin[featureBin] += output;
				CountByBin[featureBin]++;
				totalOutput += output;
			});

			SumTargetsByBin[bins.ZeroValue()] += input.SumTargets - totalOutput;
			CountByBin[bins.ZeroValue()] += input.TotalCount - bins.Count();
		}

		void SumupRootWeighted(IntArray& bins, SumupInputData& input)
		{
			if (bins.IsDense())
			{
				bins.ForEachDense([&, this](int index, int featureBin)
				{
					double output = input.Outputs[index];
					SumTargetsByBin[featureBin] += output;
					double weight = input.Weights[index];
					SumWeightsByBin[featureBin] += weight;
					CountByBin[featureBin]++;
				});
			}
			else
			{
				double totalOutput = 0.0;
				double totalWeight = 0.0;
				bins.ForEachSparse([&, this](int index, int featureBin)
				{
					double output = input.Outputs[index];
					SumTargetsByBin[featureBin] += output;
					double weight = input.Weights[index];
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

		inline void SumupLeaf(IntArray& bins, SumupInputData& input)
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

		//@TODO 貌似还是比tlc慢一些 是因为tlc的IntArray不同设置吗bit4 bit8?
		inline void SumupLeafDense(IntArray& bins, SumupInputData& input)
		{
			for (int iDocIndices = 0; iDocIndices < input.TotalCount; iDocIndices++)
			{
				double output = input.Outputs[iDocIndices];
				int index = input.DocIndices[iDocIndices];
				int featureBin = bins.values[index];
				SumTargetsByBin[featureBin] += output;
				CountByBin[featureBin]++;
			}
		}

		inline void SumupLeafSparse(IntArray& bins, SumupInputData& input)
		{
			int iDocIndices = 0;
			int totalCount = 0;
			double totalOutput = 0.0;

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
					double output = input.Outputs[iDocIndices];
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

		void SumupLeafWeighted(IntArray& bins, SumupInputData& input)
		{
			if (bins.IsDense())
			{
				for (int iDocIndices = 0; iDocIndices < input.TotalCount; iDocIndices++)
				{
					double output = input.Outputs[iDocIndices];
					double weight = input.Weights[iDocIndices];
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
				double totalOutput = 0.0;
				double totalWeight = 0.0;

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
						double output = input.Outputs[iDocIndices];
						double weight = input.Weights[iDocIndices];
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
