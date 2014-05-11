/**
 *  ==============================================================================
 *
 *          \file   LeastSquaresRegressionTreeLearner.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 19:22:20.959284
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef LEAST_SQUARES_REGRESSION_TREE_LEARNER_H_
#define LEAST_SQUARES_REGRESSION_TREE_LEARNER_H_

#include "TreeLearner.h"
#include "MappedObjectPool.h"
#include "ProbabilityFunctions.h"
#include "FeatureHistogram.h"
#include "random_util.h"

namespace gezi {

	class LeastSquaresRegressionTreeLearner : public TreeLearner
	{
	public:
		struct SplitInfo
		{
			int Feature = 0;
			uint Threshold = 0;
			double LTEOutput = 0;
			double GTOutput = 0;
			double Gain = 0;
			double GainPValue = 0;
			int LTECount = 0;
			int GTCount = 0;
		};

		struct LeafSplitCandidates
		{
			vector<SplitInfo> FeatureSplitInfo; // _bestSplitPerFeature;
			ivec DocIndices;
			int LeafIndex;
			int NumDocsInLeaf;
			double SumSquaredTargets;
			double SumTargets;
			double SumWeights;
			dvec Targets;
			dvec Weights;
			LeafSplitCandidates() = default;
			LeafSplitCandidates(int numFeatures, int numDocs, bool bHasWeights)
			{
				Init(numFeatures, numDocs, bHasWeights);
			}

			void Init(int numFeatures, int numDocs, bool bHasWeights)
			{
				FeatureSplitInfo.resize(numFeatures);
				Clear();
				DocIndices.resize(numDocs);
				Targets.resize(numDocs);
				if (bHasWeights)
				{
					Weights.resize(numDocs);
				}
			}

			void Clear()
			{
				LeafIndex = -1;
				for (size_t f = 0; f < FeatureSplitInfo.size(); f++)
				{
					FeatureSplitInfo[f].Feature = f;
					FeatureSplitInfo[f].Gain = -std::numeric_limits<double>::infinity();
				}
			}

			void Initialize()
			{
				Clear();
			}

			void Initialize(const dvec& targets, const dvec& weights, bool filterZeros)
			{
				Clear();
				SumTargets = 0.0;
				SumWeights = 0.0;
				SumSquaredTargets = 0.0;
				LeafIndex = 0;
				NumDocsInLeaf = targets.size();
				if (filterZeros)
				{
					int nonZeroCount = 0;
					for (int i = 0; i < NumDocsInLeaf; i++)
					{
						double target = targets[i];
						if (target != 0.0)
						{
							Targets[nonZeroCount] = target;
							SumTargets += target;
							DocIndices[nonZeroCount] = i;
							nonZeroCount++;
							if (!Weights.empty())
							{
								double weight = weights[i];
								Weights[nonZeroCount] = weight;
								SumWeights += weight;
								if (weight != 0.0)
								{
									SumSquaredTargets += (target * target) / weight;
								}
							}
							else
							{
								SumSquaredTargets += target * target;
							}
						}
					}
					NumDocsInLeaf = nonZeroCount;
				}
				else
				{
					DocIndices.clear();
					for (int i = 0; i < NumDocsInLeaf; i++)
					{
						double target = targets[i];
						Targets[i] = target;
						SumTargets += target;
						if (!Weights.empty())
						{
							double weight = weights[i];
							Weights[i] = weight;
							SumWeights += weight;
							if (weight != 0.0)
							{
								SumSquaredTargets += (target * target) / weight;
							}
						}
						else
						{
							SumSquaredTargets += target * target;
						}
					}
				}
			}

			void Initialize(int leafIndex, DocumentPartitioning& partitioning, const dvec& targets, const dvec& weights, bool filterZeros)
			{
				Clear();
				SumTargets = 0.0;
				SumWeights = 0.0;
				SumSquaredTargets = 0.0;
				LeafIndex = leafIndex;
				NumDocsInLeaf = partitioning.GetLeafDocuments(leafIndex, DocIndices);
				int nonZeroCount = 0;
				for (int i = 0; i < NumDocsInLeaf; i++)
				{
					int docIndex = DocIndices[i];
					double target = targets[docIndex];
					if ((target != 0.0) || !filterZeros)
					{
						Targets[nonZeroCount] = target;
						SumTargets += target;
						DocIndices[nonZeroCount] = docIndex;
						if (!Weights.empty())
						{
							double weight = weights[docIndex];
							Weights[nonZeroCount] = weight;
							SumWeights += weight;
							if (weight != 0.0)
							{
								SumSquaredTargets += (target * target) / weight;
							}
						}
						else
						{
							SumSquaredTargets += target * target;
						}
						nonZeroCount++;
					}
				}
				NumDocsInLeaf = nonZeroCount;
			}

			void Initialize(int leafIndex, const ivec& docIndices, int begin, int length, const dvec& targets, bool filterZeros)
			{
				Clear();
				SumTargets = 0.0;
				LeafIndex = leafIndex;
				fill_range(DocIndices.begin(), docIndices.begin(), length);
				NumDocsInLeaf = length;
				int nonZeroCount = 0;
				for (int i = 0; i < NumDocsInLeaf; i++)
				{
					double target = targets[DocIndices[i]];
					if ((target != 0.0) || !filterZeros)
					{
						Targets[nonZeroCount] = target;
						SumTargets += target;
						DocIndices[nonZeroCount] = DocIndices[i];
						nonZeroCount++;
					}
				}
				NumDocsInLeaf = nonZeroCount;
			}

			double VarianceTargets()
			{
				double denom = (Weights.empty()) ? ((double)NumDocsInLeaf) : SumWeights;
				return ((SumSquaredTargets - (SumTargets / denom)) / (denom - 1.0));
			}

		};

	protected:
		BitArray _activeFeatures;
		bool _allowDummies;
		bool _areTargetsWeighted = false;
		vector<SplitInfo> _bestSplitInfoPerLeaf;
		double _bsrMaxTreeOutput;
		double _entropyCoefficient;
		double _featureFirstUsePenalty;
		double _featureReusePenalty;
		ivec _featureUseCount;
		bool _filterZeros;
		double _gainConfidenceInSquaredStandardDeviations;
		int _minDocsInLeaf;
		int _numLeaves;
		Random _rand;
		MappedObjectPool<vector<FeatureHistogram> > _histogramArrayPool;
		vector<FeatureHistogram> _parentHistogramArray;
		vector<FeatureHistogram> _largerChildHistogramArray;
		LeafSplitCandidates _largerChildSplitCandidates;
		vector<FeatureHistogram> _smallerChildHistogramArray;
		LeafSplitCandidates _smallerChildSplitCandidates;
		double _softmaxTemperature;
		double _splitFraction;
	public:
		LeastSquaresRegressionTreeLearner(Dataset& trainData, int numLeaves, int minDocsInLeaf, double entropyCoefficient, double featureFirstUsePenalty, double featureReusePenalty, double softmaxTemperature, int histogramPoolSize, int randomSeed, double splitFraction, bool filterZeros)
			: LeastSquaresRegressionTreeLearner(trainData, numLeaves, minDocsInLeaf, entropyCoefficient, featureFirstUsePenalty, featureReusePenalty, softmaxTemperature, histogramPoolSize, randomSeed, splitFraction, filterZeros, false, 0.0, false, -1.0)
		{
		}

		LeastSquaresRegressionTreeLearner(Dataset& trainData, int numLeaves, int minDocsInLeaf, double entropyCoefficient, double featureFirstUsePenalty, double featureReusePenalty, double softmaxTemperature, int histogramPoolSize, int randomSeed, double splitFraction, bool filterZeros, bool allowDummies, double gainConfidenceLevel, bool areTargetsWeighted, double bsrMaxTreeOutput)
			: TreeLearner(trainData, numLeaves), _rand(randomSeed)
		{
			_minDocsInLeaf = minDocsInLeaf;
			_allowDummies = allowDummies;
			_entropyCoefficient = entropyCoefficient * 1E-06;
			_featureFirstUsePenalty = featureFirstUsePenalty;
			_featureReusePenalty = featureReusePenalty;
			_softmaxTemperature = softmaxTemperature;
			_areTargetsWeighted = areTargetsWeighted;
			_bestSplitInfoPerLeaf.resize(numLeaves);

			vector<vector<FeatureHistogram> > histogramPool(histogramPoolSize);
			for (int i = 0; i < histogramPoolSize; i++)
			{
				histogramPool[i].resize(TrainData.NumFeatures);
				for (int j = 0; j < TrainData.NumFeatures; j++)
				{
					histogramPool[i][j].Initialize(TrainData.Features[j], HasWeights());
				}
				/*	for (int j = 0; j < TrainData.NumFeatures; j++)
					{
					histogramPool[i].emplace_back(FeatureHistogram(TrainData.Features[j], HasWeights()));
					}*/
			}
			_histogramArrayPool.Initialize(histogramPool, numLeaves - 1);

			MakeSplitCandidateArrays(TrainData.NumFeatures, TrainData.NumDocs);
			_featureUseCount.resize(TrainData.NumFeatures);
			_splitFraction = splitFraction;
			_filterZeros = filterZeros;
			_bsrMaxTreeOutput = bsrMaxTreeOutput;
			_gainConfidenceInSquaredStandardDeviations = ProbabilityFunctions::Probit(1.0 - ((1.0 - gainConfidenceLevel) * 0.5));
			_gainConfidenceInSquaredStandardDeviations *= _gainConfidenceInSquaredStandardDeviations;
		}


		virtual RegressionTree FitTargets(dvec& targets) override
		{
			int maxLeaves = NumLeaves;
			int LTEChild;
			int GTChild;
			Initialize();
			RegressionTree tree = NewTree();
			SetRootModel(tree, targets);
			FindBestSplitOfRoot(targets);
			int bestLeaf = 0;
			SplitInfo rootSplitInfo = _bestSplitInfoPerLeaf[0];
			if (rootSplitInfo.Gain == -std::numeric_limits<double>::infinity())
			{
				if (!_allowDummies)
				{
					THROW((format("Learner cannot build a tree with root split gain = %lf, dummy splits disallowed") % rootSplitInfo.Gain).str());
				}
				LOG(WARNING) << "Learner cannot build a tree with root split gain = " << rootSplitInfo.Gain << ", so a dummy tree will be used instead";
				double rootTarget = _smallerChildSplitCandidates.SumTargets / ((double)_smallerChildSplitCandidates.NumDocsInLeaf);
				MakeDummyRootSplit(tree, rootTarget, targets);
				return tree;
			}
			_featureUseCount[rootSplitInfo.Feature]++;
			PerformSplit(tree, 0, targets, LTEChild, GTChild);
			for (int split = 0; split < (maxLeaves - 2); split++)
			{
				FindBestSplitOfSiblings(LTEChild, GTChild, Partitioning, targets);
				bestLeaf = GetBestFeature(_bestSplitInfoPerLeaf);
				SplitInfo bestLeafSplitInfo = _bestSplitInfoPerLeaf[bestLeaf];
				if (bestLeafSplitInfo.Gain <= 0.0)
				{
					LOG(WARNING) << "We cannot perform more splits with gain = " << bestLeafSplitInfo.Gain;
					break;
				}
				_featureUseCount[bestLeafSplitInfo.Feature]++;
				PerformSplit(tree, bestLeaf, targets, LTEChild, GTChild);
			}
			return tree;
		}
	protected:
		bool HasWeights()
		{
			return _areTargetsWeighted;
		}

		void Initialize()
		{
			_histogramArrayPool.Reset();
			Partitioning.Initialize();
		}

		void MakeDummyRootSplit(RegressionTree& tree, double rootTarget, dvec& targets)
		{
			int dummyLTEChild;
			int dummyGTChild;
			SplitInfo newRootSplitInfo;
			newRootSplitInfo.LTEOutput = rootTarget;
			newRootSplitInfo.GTOutput = rootTarget;
			_bestSplitInfoPerLeaf[0] = newRootSplitInfo;
			PerformSplit(tree, 0, targets, dummyLTEChild, dummyGTChild);
		}

		void MakeSplitCandidateArrays(int numFeatures, int numDocs)
		{
			_smallerChildSplitCandidates.Init(numFeatures, numDocs, HasWeights());
			_largerChildSplitCandidates.Init(numFeatures, numDocs, HasWeights());
		}

		RegressionTree NewTree()
		{
			return RegressionTree(NumLeaves);
		}

		void PerformSplit(RegressionTree& tree, int bestLeaf, dvec& targets,
			int& LTEChild, int& GTChild)
		{
			SplitInfo bestSplitInfo = _bestSplitInfoPerLeaf[bestLeaf];
			int newInteriorNodeIndex = tree.Split(bestLeaf, bestSplitInfo.Feature, bestSplitInfo.Threshold, bestSplitInfo.LTEOutput, bestSplitInfo.GTOutput, bestSplitInfo.Gain, bestSplitInfo.GainPValue);
			GTChild = ~tree.GTChild(newInteriorNodeIndex);
			LTEChild = bestLeaf;
			Partitioning.Split(bestLeaf, TrainData.Features[bestSplitInfo.Feature].Bins, bestSplitInfo.Threshold, GTChild);
		}

		void SetBestFeatureForLeaf(LeafSplitCandidates leafSplitCandidates, int bestFeature)
		{
			int leaf = leafSplitCandidates.LeafIndex;
			_bestSplitInfoPerLeaf[leaf] = leafSplitCandidates.FeatureSplitInfo[bestFeature];
			_bestSplitInfoPerLeaf[leaf].Feature = bestFeature;
		}

		double CalculateSplittedLeafOutput(int totalCount, double sumTargets, double sumWeights)
		{
			if (!HasWeights())
			{
				return (sumTargets / ((double)totalCount));
			}
			if (_bsrMaxTreeOutput < 0.0)
			{
				return (sumTargets / sumWeights);
			}
			return (sumTargets / (2.0 * sumWeights));
		}

		int GetBestFeature(vector<SplitInfo>& featureSplitInfo)
		{
			return 	max_element(featureSplitInfo.begin(), featureSplitInfo.end(), 
				[](SplitInfo& l, SplitInfo& r) {return l.Gain < r.Gain; }) - featureSplitInfo.begin();
		}

		void FindAndSetBestFeatureForLeaf(LeafSplitCandidates& leafSplitCandidates)
		{
			int bestFeature = GetBestFeature(leafSplitCandidates.FeatureSplitInfo);
			SetBestFeatureForLeaf(leafSplitCandidates, bestFeature);
		}

		dvec GetTargetWeights()
		{
			//@TODO
			//return TrainData.GetDatasetSkeleton.GetData<double>(TreeLearner.TargetWeightsDatasetName);
			dvec result;
			return result;
		}

		double GetLeafSplitGain(int count, double starget, double sweight)
		{
			if (!HasWeights())
			{
				return ((starget * starget) / ((double)count));
			}
			double astarget = std::abs(starget);
			if ((_bsrMaxTreeOutput > 0.0) && (astarget >= ((2.0 * sweight) * _bsrMaxTreeOutput)))
			{
				return ((4.0 * _bsrMaxTreeOutput) * (astarget - (_bsrMaxTreeOutput * sweight)));
			}
			return ((starget * starget) / sweight);
		}

		void FindBestSplitOfRoot(dvec& targets)
		{
			if (Partitioning.NumDocs() == TrainData.NumDocs)
			{
				_smallerChildSplitCandidates.Initialize(targets, GetTargetWeights(), _filterZeros);
			}
			else
			{
				_smallerChildSplitCandidates.Initialize(0, Partitioning, targets, GetTargetWeights(), _filterZeros);
			}
			_parentHistogramArray.clear();
			_histogramArrayPool.Get(0, _smallerChildHistogramArray);
			_largerChildSplitCandidates.Initialize();
#pragma omp parallel for
			for (int featureIndex = 0; featureIndex < TrainData.NumFeatures; featureIndex++)
			{
				FindBestThresholdForFeature(featureIndex);
			}
			FindAndSetBestFeatureForLeaf(_smallerChildSplitCandidates);
		}

		void FindBestSplitOfSiblings(int LTEChild, int GTChild, DocumentPartitioning& partitioning, const dvec& targets)
		{
			int numDocsInLTEChild = partitioning.NumDocsInLeaf(LTEChild);
			int numDocsInGTChild = partitioning.NumDocsInLeaf(GTChild);
			if ((numDocsInGTChild < (_minDocsInLeaf * 2)) && (numDocsInLTEChild < (_minDocsInLeaf * 2)))
			{
				_bestSplitInfoPerLeaf[LTEChild].Gain = -std::numeric_limits<double>::infinity();
				_bestSplitInfoPerLeaf[GTChild].Gain = -std::numeric_limits<double>::infinity();
			}
			else
			{
				_parentHistogramArray.clear();
				if (numDocsInLTEChild < numDocsInGTChild)
				{
					_smallerChildSplitCandidates.Initialize(LTEChild, partitioning, targets, GetTargetWeights(), _filterZeros);
					_largerChildSplitCandidates.Initialize(GTChild, partitioning, targets, GetTargetWeights(), _filterZeros);
					if (_histogramArrayPool.Get(LTEChild, _largerChildHistogramArray))
					{
						_parentHistogramArray = _largerChildHistogramArray;
					}
					_histogramArrayPool.Steal(LTEChild, GTChild);
					_histogramArrayPool.Get(LTEChild, _smallerChildHistogramArray);
				}
				else
				{
					_smallerChildSplitCandidates.Initialize(GTChild, partitioning, targets, GetTargetWeights(), _filterZeros);
					_largerChildSplitCandidates.Initialize(LTEChild, partitioning, targets, GetTargetWeights(), _filterZeros);
					if (_histogramArrayPool.Get(LTEChild, _largerChildHistogramArray))
					{
						_parentHistogramArray = _largerChildHistogramArray;
					}
					_histogramArrayPool.Get(GTChild, _smallerChildHistogramArray);
				}
#pragma omp parallel for
				for (int featureIndex = 0; featureIndex < TrainData.NumFeatures; featureIndex++)
				{
					FindBestThresholdForFeature(featureIndex);
				}
				FindAndSetBestFeatureForLeaf(_smallerChildSplitCandidates);
				FindAndSetBestFeatureForLeaf(_largerChildSplitCandidates);
			}
		}

		void FindBestThresholdForFeature(int featureIndex)
		{
			if ((!_parentHistogramArray.empty()) && !_parentHistogramArray[featureIndex].IsSplittable)
			{
				_smallerChildHistogramArray[featureIndex].IsSplittable = false;
			}
			else
			{
				_smallerChildHistogramArray[featureIndex].SumupWeighted(featureIndex, _smallerChildSplitCandidates.NumDocsInLeaf, _smallerChildSplitCandidates.SumTargets, _smallerChildSplitCandidates.SumWeights, _smallerChildSplitCandidates.Targets, _smallerChildSplitCandidates.Weights, _smallerChildSplitCandidates.DocIndices);
				FindBestThresholdFromHistogram(_smallerChildHistogramArray[featureIndex], _smallerChildSplitCandidates, featureIndex);
				if (_largerChildSplitCandidates.LeafIndex >= 0)
				{
					//or affine tree
					if (_parentHistogramArray.empty())
					{
						_largerChildHistogramArray[featureIndex].SumupWeighted(featureIndex, _largerChildSplitCandidates.NumDocsInLeaf, _largerChildSplitCandidates.SumTargets, _largerChildSplitCandidates.SumWeights, _largerChildSplitCandidates.Targets, _largerChildSplitCandidates.Weights, _largerChildSplitCandidates.DocIndices);
					}
					else
					{
						_largerChildHistogramArray[featureIndex].Subtract(_smallerChildHistogramArray[featureIndex]);
					}
					FindBestThresholdFromHistogram(_largerChildHistogramArray[featureIndex], _largerChildSplitCandidates, featureIndex);
				}
			}
		}

		void FindBestThresholdFromHistogram(FeatureHistogram& histogram, LeafSplitCandidates& leafSplitCandidates, int feature)
		{
			double bestSumLTETargets = std::numeric_limits<double>::quiet_NaN();
			double bestSumLTEWeights = std::numeric_limits<double>::quiet_NaN();
			double bestShiftedGain = -std::numeric_limits<double>::infinity();
			double trust = TrainData.Features[feature].Trust;
			int bestLTECount = -1;
			uint bestThreshold = (uint)histogram.NumFeatureValues;
			double eps = 1E-10;
			double sumLTETargets = 0.0;
			double sumLTEWeights = eps;
			int LTECount = 0;
			int totalCount = leafSplitCandidates.NumDocsInLeaf;
			double sumTargets = leafSplitCandidates.SumTargets;
			double sumWeights = leafSplitCandidates.SumWeights + (2.0 * eps);
			double gainShift = GetLeafSplitGain(totalCount, sumTargets, sumWeights);
			double minShiftedGain = (_gainConfidenceInSquaredStandardDeviations <= 0.0) ? 0.0 : ((((_gainConfidenceInSquaredStandardDeviations * leafSplitCandidates.VarianceTargets()) * totalCount) / ((double)(totalCount - 1))) + gainShift);
			histogram.IsSplittable = false;
			double minDocsForThis = ((double)_minDocsInLeaf) / trust;
			for (int t = 0; t < (histogram.NumFeatureValues - 1); t += 1)
			{
				sumLTETargets += histogram.SumTargetsByBin[t];
				if (!histogram.SumWeightsByBin.empty())
				{
					sumLTEWeights += histogram.SumWeightsByBin[t];
				}
				LTECount += histogram.CountByBin[t];
				if (LTECount >= minDocsForThis)
				{
					int GTCount = totalCount - LTECount;
					if (GTCount < minDocsForThis)
					{
						break;
					}
					double sumGTTargets = sumTargets - sumLTETargets;
					double sumGTWeights = sumWeights - sumLTEWeights;
					double currentShiftedGain = GetLeafSplitGain(LTECount, sumLTETargets, sumLTEWeights) + GetLeafSplitGain(GTCount, sumGTTargets, sumGTWeights);
					if (currentShiftedGain >= minShiftedGain)
					{
						histogram.IsSplittable = true;
						if (_entropyCoefficient > 0.0)
						{
							double entropyGain = ((totalCount * std::log((double)totalCount)) - (LTECount * std::log((double)LTECount))) - (GTCount * std::log((double)GTCount));
							currentShiftedGain += _entropyCoefficient * entropyGain;
						}
						if (currentShiftedGain > bestShiftedGain)
						{
							bestLTECount = LTECount;
							bestSumLTETargets = sumLTETargets;
							bestSumLTEWeights = sumLTEWeights;
							bestThreshold = t;
							bestShiftedGain = currentShiftedGain;
						}
					}
				}
			}
			leafSplitCandidates.FeatureSplitInfo[feature].Threshold = bestThreshold;
			leafSplitCandidates.FeatureSplitInfo[feature].LTEOutput = CalculateSplittedLeafOutput(bestLTECount, bestSumLTETargets, bestSumLTEWeights);
			leafSplitCandidates.FeatureSplitInfo[feature].GTOutput = CalculateSplittedLeafOutput(totalCount - bestLTECount, sumTargets - bestSumLTETargets, sumWeights - bestSumLTEWeights);
			double usePenalty = (_featureUseCount[feature] == 0) ? _featureFirstUsePenalty : (_featureReusePenalty * std::log((double)(_featureUseCount[feature] + 1)));
			leafSplitCandidates.FeatureSplitInfo[feature].Gain = ((bestShiftedGain - gainShift) * trust) - usePenalty;
			double erfcArg = std::sqrt(((bestShiftedGain - gainShift) * (totalCount - 1)) / ((2.0 * leafSplitCandidates.VarianceTargets()) * totalCount));
			leafSplitCandidates.FeatureSplitInfo[feature].GainPValue = ProbabilityFunctions::Erfc(erfcArg);
		}

		void SetRootModel(RegressionTree& tree, dvec& targets)
		{
		}
	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of LEAST_SQUARES_REGRESSION_TREE_LEARNER_H_
