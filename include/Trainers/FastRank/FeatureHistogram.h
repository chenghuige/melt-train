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
#include "Feature->h"
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
			if (numBins > sumTarget.size())
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
			if (feature->NumBins() > CountByBin.size())
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
			dvec BinMedians;
			ivec DocIndices;
			dvec Outputs;
			double SumTargets;
			double SumWeights;
			int TotalCount;
			dvec Weights;
			SumupInputData(int TotalCount, double SumTargets, double SumWeights, dvec& Outputs,
				dvec& Weights, ivec& DocIndices, dvec& BinMedians)
			{
				TotalCount = TotalCount;
				SumTargets = SumTargets;
				Outputs = Outputs;
				DocIndices = DocIndices;
				BinMedians = BinMedians;
				SumWeights = SumWeights;
				Weights = Weights;
			}
		};

		void Sumup(int featureIndex, int numDocsInLeaf, double sumTargets, 
			dvec& outputs, ivec& docIndices)
		{
			dvec weights;
			SumupWeighted(featureIndex, numDocsInLeaf, sumTargets, 0.0, outputs, weights, docIndices);
		}

		void SumupWeighted(int featureIndex, int numDocsInLeaf, double sumTargets, double sumWeights,
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
		void Sumup(IntArray& bins, SumupInputData& input)
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

		void SumupRoot(IntArray& bins, SumupInputData& input)
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

		void SumupRootDense(IntArray& bins, SumupInputData& input)
		{
			bins.ForEachDense([&, this](int index, int featureBin)
			{
				double output = input.Outputs[index];
				SumTargetsByBin[featureBin] += output;
				CountByBin[featureBin]++;
			});
		}

		void SumupRootSparse(IntArray& bins, SumupInputData& input)
		{
			double totalOutput = 0.0;
			int currentPos = 0;
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

		//@TODO
		void SumupRootWeighted(IntArray& bins, SumupInputData& input)
		{

		}

		void SumupLeaf(IntArray& bins, SumupInputData& input)
		{
			if (bins.IsDense())
			{
				SumupLeafDense(bins, input);
			}
			else
			{
				SumupLeafSparse(bins, input);
			}
		}

		void SumupLeafDense(IntArray& bins, SumupInputData& input)
		{
			int iDocIndices = 0;
			bool ok = true;
			bins.ForEachDenseIf([&, this](int index, int featureBin, bool& isOk)
			{
				if (index == input.DocIndices[iDocIndices])
				{
					double output = input.Outputs[iDocIndices * 8]; //@TODO why * 8
					SumTargetsByBin[featureBin] += output;
					CountByBin[featureBin]++;
					iDocIndices++;
					if (iDocIndices >= input.TotalCount)
					{
						isOk = false;
					}
				}
				else if (index > input.DocIndices[iDocIndices])
				{
					iDocIndices++;
					if (iDocIndices >= input.TotalCount)
					{
						isOk = false;
					}
				}
			}, ok);
		}
		void SumupLeafSparse(IntArray& bins, SumupInputData& input)
		{
			int iDocIndices = 0;
			int totalCount = 0;
			double totalOutput = 0.0;
			bool ok = true;
			bins.ForEachSparseIf([&, this](int index, int featureBin, bool& isOk)
			{
				if (index == input.DocIndices[iDocIndices])
				{
					double output = input.Outputs[iDocIndices]; 
					SumTargetsByBin[featureBin] += output;
					totalOutput += output;
					CountByBin[featureBin]++;
					totalCount++;
					iDocIndices++;
					if (iDocIndices >= input.TotalCount)
					{
						isOk = false;
					}
				}
				else if (index > input.DocIndices[iDocIndices])
				{
					iDocIndices++;
					if (iDocIndices >= input.TotalCount)
					{
						isOk = false;
					}
				}
			}, ok);
			SumTargetsByBin[bins.ZeroValue()] += input.SumTargets - totalOutput;
			CountByBin[bins.ZeroValue()] += input.TotalCount - bins.Count();
		}

		//@TODO
		void SumupLeafWeighted(IntArray& bins, SumupInputData& input)
		{

		}
	};

}  //----end of namespace gezi

#endif  //----end of FEATURE_HISTOGRAM_H_
