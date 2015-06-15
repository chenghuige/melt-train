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
#include "rabit_util.h"

namespace gezi {

	class LeastSquaresRegressionTreeLearner : public TreeLearner
	{
	public:
		struct SplitInfo
		{
			int Feature = 0;
			uint Threshold = 0;
			Float LTEOutput = 0;
			Float GTOutput = 0;
			Float Gain = -std::numeric_limits<Float>::infinity();
			Float GainPValue = -std::numeric_limits<Float>::infinity();
			int LTECount = 0;
			int GTCount = 0;

			void Clear()
			{
				Feature = 0;
				Threshold = 0;
				LTEOutput = 0;
				GTOutput = 0;
				Gain = -std::numeric_limits<Float>::infinity();
				GainPValue = -std::numeric_limits<Float>::infinity();
				LTECount = 0;
				GTCount = 0;
			}

			inline static void Reduce(SplitInfo& dest, const SplitInfo& src)
			{
				if (src.Gain > dest.Gain)
				{
					dest = src;
				}
			}
		};



		struct LeafSplitCandidates
		{
			vector<SplitInfo> FeatureSplitInfo; // _bestSplitPerFeature; ��¼������ÿ��Feature�ķ�����Ϣ
			ivec DocIndices;
			ivec _docIndicesCopy;
			int LeafIndex = -1;
			int NumDocsInLeaf;
			Float SumSquaredTargets = 0;
			Float SumTargets = 0;
			Float SumWeights = 0;
			Fvec Targets;
			Fvec Weights;
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
					FeatureSplitInfo[f].Gain = -std::numeric_limits<Float>::infinity();
				}
			}

			void Initialize()
			{
				Clear();
			}

			void Initialize(const Fvec& targets, const Fvec& weights, bool filterZeros)
			{
				Clear();
				SumTargets = 0.0;
				SumWeights = 0.0;
				SumSquaredTargets = 0.0;
				LeafIndex = 0;
				NumDocsInLeaf = targets.size();
				if (filterZeros)
				{
					if (DocIndices.empty())
						DocIndices.swap(_docIndicesCopy);
					int nonZeroCount = 0;
					for (int i = 0; i < NumDocsInLeaf; i++)
					{
						Float target = targets[i];
						if (target != 0.0)
						{
							Targets[nonZeroCount] = target;
							SumTargets += target;
							DocIndices[nonZeroCount] = i;
							nonZeroCount++;
							if (!Weights.empty())
							{
								Float weight = weights[i];
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
					//��ǰ_filterZero=false�ߵ����·�� DocIndices���empty,��������˶�Ӧ����doc����target֮��
					//��DocIndices���empty��������FeatureHistogram�����Ǳ�ʶ��root��sumup,root��ζ�Ŷ������ĵ���������ǲ���(Ҷ���ڲ�) ����SumupRoot��SumupLeaf������
					//ͬʱ����ʼ��ʹ��DocIndices���ڲ�ͨ��_docIndicesCopy���������ظ����ٿռ�
					if (!DocIndices.empty())
						_docIndicesCopy.swap(DocIndices);
					for (int i = 0; i < NumDocsInLeaf; i++)
					{
						Float target = targets[i];
						Targets[i] = target;
						SumTargets += target;
						if (!Weights.empty())
						{
							Float weight = weights[i];
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

			void Initialize(int leafIndex, DocumentPartitioning& partitioning, const Fvec& targets, const Fvec& weights, bool filterZeros)
			{
				Clear();
				SumTargets = 0.0;
				SumWeights = 0.0;
				SumSquaredTargets = 0.0;
				LeafIndex = leafIndex;
				if (DocIndices.empty())
					DocIndices.swap(_docIndicesCopy);
				NumDocsInLeaf = partitioning.GetLeafDocuments(leafIndex, DocIndices);
				int nonZeroCount = 0;
				for (int i = 0; i < NumDocsInLeaf; i++)
				{
					int docIndex = DocIndices[i];
					Float target = targets[docIndex];
					if ((target != 0.0) || !filterZeros)
					{
						Targets[nonZeroCount] = target;
						SumTargets += target;
						DocIndices[nonZeroCount] = docIndex;
						if (!Weights.empty())
						{
							Float weight = weights[docIndex];
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

			void Initialize(int leafIndex, const ivec& docIndices, int begin, int length, const Fvec& targets, bool filterZeros)
			{
				Clear();
				SumTargets = 0.0;
				LeafIndex = leafIndex;
				if (DocIndices.empty())
					DocIndices.swap(_docIndicesCopy);
				gezi::fill_range(DocIndices.begin(), docIndices.begin(), length);
				NumDocsInLeaf = length;
				int nonZeroCount = 0;
				for (int i = 0; i < NumDocsInLeaf; i++)
				{
					Float target = targets[DocIndices[i]];
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

			Float VarianceTargets()
			{
				Float denom = (Weights.empty()) ? ((Float)NumDocsInLeaf) : SumWeights;
				return ((SumSquaredTargets - (SumTargets / denom)) / (denom - 1.0));
			}

		};

	protected:
		bool _allowDummies;
		bool _areTargetsWeighted = false;

		Float _bsrMaxTreeOutput;
		Float _entropyCoefficient;
		Float _featureFirstUsePenalty;
		Float _featureReusePenalty;
		ivec _featureUseCount;
		bool _filterZeros;
		Float _gainConfidenceInSquaredStandardDeviations;
		int _minDocsInLeaf;
		int _numLeaves;
		Random _rand;
		//�洢ÿ���ڲ��ڵ��Ӧÿ��Feature��Histogram��Ϣ,Ҳ��������Sum��Ϣ �������ѵ�ʱ����Լ����Ż�
		//ֻ��Ҫ�������Ҷ�ӵĲ��ֵ�HistogramȻ�������parent�ڲ��ڵ��Histogram��Ϣ��ȥ���Histogram�õ��϶�Ҷ�Ӳ��ֵ�Histogram
		MappedObjectPool<vector<FeatureHistogram> > _histogramArrayPool;
		vector<FeatureHistogram>* _parentHistogramArray = NULL;
		vector<FeatureHistogram>* _largerChildHistogramArray = NULL;  //�϶�Instance�ĺ���Histogram��Ϣ
		vector<FeatureHistogram>* _smallerChildHistogramArray = NULL; //����Instance�ĺ���Histogram��Ϣ
		//��¼�ۺ�����_smallerChildSplitCandidates,_largerChildSplitCandidates����֮�����ѷ�����Ϣ
		vector<SplitInfo> _bestSplitInfoPerLeaf;
		LeafSplitCandidates _smallerChildSplitCandidates; //��������Ϣ
		LeafSplitCandidates _largerChildSplitCandidates;  //��������Ϣ
		Float _softmaxTemperature;
		Float _splitFraction;
		bool _preSplitCheck = false;
	public:
		LeastSquaresRegressionTreeLearner(Dataset& trainData, int numLeaves, int minDocsInLeaf, Float entropyCoefficient,
			Float featureFirstUsePenalty, Float featureReusePenalty, Float softmaxTemperature, int histogramPoolSize,
			int randomSeed, Float splitFraction, bool preSplitCheck, bool filterZeros, bool allowDummies,
			Float gainConfidenceLevel, bool areTargetsWeighted, Float bsrMaxTreeOutput)
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

			//histogramPoolSize ��ps�������� ���������ڲ��ڵ���Ŀ����������ȫ����Ϣ�����棬��ѣ�
			//ͨ��MappedObjectPool������Խ�Լ�ڴ棬Ĭ������2/3*�ڲ��ڵ���Ŀ���ٶȼ�����ͬ
			vector<vector<FeatureHistogram> > histogramPool(histogramPoolSize);
			for (int i = 0; i < histogramPoolSize; i++)
			{
				histogramPool[i].resize(TrainData.NumFeatures);
				for (int j = 0; j < TrainData.NumFeatures; j++)
				{
					if (!Rabit::Choose(j))
						continue;
					histogramPool[i][j].Initialize(TrainData.Features[j], HasWeights());
				}
			}
			_histogramArrayPool.Initialize(histogramPool, numLeaves - 1); //��Ҫ����Ĵ�СΪ�����ڲ��ڵ����

			MakeSplitCandidateArrays(TrainData.NumFeatures, TrainData.NumDocs);
			_featureUseCount.resize(TrainData.NumFeatures, 0);
			_splitFraction = splitFraction;
			_preSplitCheck = preSplitCheck;
			_filterZeros = filterZeros;
			_bsrMaxTreeOutput = bsrMaxTreeOutput;
			_gainConfidenceInSquaredStandardDeviations = ProbabilityFunctions::Probit(1.0 - ((1.0 - gainConfidenceLevel) * 0.5));
			_gainConfidenceInSquaredStandardDeviations *= _gainConfidenceInSquaredStandardDeviations;
			PVAL(_gainConfidenceInSquaredStandardDeviations);
		}

		virtual RegressionTree FitTargets(const BitArray& activeFeatures, Fvec& targets) override
		{
			Initialize(activeFeatures);
			RegressionTree tree = NewTree();
			SetRootModel(tree, targets);
			FindBestSplitOfRoot(targets);
			const SplitInfo& rootSplitInfo = _bestSplitInfoPerLeaf[0];
			if (rootSplitInfo.Gain == -std::numeric_limits<Float>::infinity())
			{
				CHECK(_allowDummies) << format("Learner cannot build a tree with root split gain = {:lf}, dummy splits disallowed", rootSplitInfo.Gain);
				LOG(WARNING) << "Learner cannot build a tree with root split gain = " << rootSplitInfo.Gain << ", so a dummy tree will be used instead";
				Float rootTarget = _smallerChildSplitCandidates.SumTargets / ((Float)_smallerChildSplitCandidates.NumDocsInLeaf);
				MakeDummyRootSplit(tree, rootTarget, targets);
				return tree;
			}
			_featureUseCount[rootSplitInfo.Feature]++;
			int LTEChild, GTChild;
			PerformSplit(tree, 0, targets, LTEChild, GTChild);
			for (int split = 1; split < (NumLeaves - 1); split++)
			{
				FindBestSplitOfSiblings(LTEChild, GTChild, Partitioning, targets);
				//FindBestSplitOfSiblingsSimple(LTEChild, GTChild, Partitioning, targets);
				int bestLeaf = GetBestSplit(_bestSplitInfoPerLeaf);
				const SplitInfo& bestLeafSplitInfo = _bestSplitInfoPerLeaf[bestLeaf];
				//PrintVecTopN(_bestSplitInfoPerLeaf, Gain, 10);
				//if (bestLeafSplitInfo.Gain <= 0.0)
				if (bestLeafSplitInfo.Gain < std::numeric_limits<Float>::epsilon()) // <= 0
				{
					VLOG(6) << "We cannot perform more splits with gain = " << bestLeafSplitInfo.Gain << " in split " << split;
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

		virtual bool IsFeatureOk(int index) override
		{
			//@TODO ��һ��򵥳��԰��������ָ�,�����ڴ�ռ�ò����� ֻ�Ǽ��ټ��������������Ը�Ϊֻ�洢�Լ������������Ϣ
			if (!Rabit::Choose(index))
			{
				return false;
			}
			if (!_preSplitCheck)
			{
				return (*_activeFeatures)[index];
			}
			else
			{ //ǰ�ڼ������������������@FIXME  ���߳�rand ������
				return (*_activeFeatures)[index] && _rand.NextDouble() > _splitFraction;
			}
		}

		//@TODO ���ղ�һ�µ�΢С����ܿ����������� Ū����Ƿ���Ҫclear �Ƿ���ָ����Ա������⣿
		void ClearBestSplitInfos()
		{
			for (auto& info : _bestSplitInfoPerLeaf)
			{
				info.Clear();
			}
		}
		void Initialize(const BitArray& activeFeatures)
		{
			_activeFeatures = &activeFeatures;
			_histogramArrayPool.Reset();
			Partitioning.Initialize();

			ClearBestSplitInfos();
		}

		void MakeDummyRootSplit(RegressionTree& tree, Float rootTarget, Fvec& targets)
		{
			int dummyLTEChild;
			int dummyGTChild;

			_bestSplitInfoPerLeaf[0].LTEOutput = rootTarget;
			_bestSplitInfoPerLeaf[0].GTOutput = rootTarget;
			//----------û������� ��ʹ������Ҷ�ӷֵ�ͬһ���ڵ� @?
			_bestSplitInfoPerLeaf[0].Feature = 0;
			_bestSplitInfoPerLeaf[0].Threshold = 0;
			_bestSplitInfoPerLeaf[0].Gain = 0;

			PerformSplit(tree, 0, targets, dummyLTEChild, dummyGTChild);
		}

		void MakeSplitCandidateArrays(int numFeatures, int numDocs)
		{
			_smallerChildSplitCandidates.Init(numFeatures, numDocs, HasWeights());
			_largerChildSplitCandidates.Init(numFeatures, numDocs, HasWeights());
		}

		RegressionTree NewTree()
		{
			return RegressionTree(NumLeaves, TrainData.Features);
		}

		//Regression����Ҷ�ӽڵ����ڲ��ڵ�����µ�����Ҷ��, Partitioning��¼���ѵ�doc��Ϣa
		//doc���������У�leaf��Ӧ������¼����ʼλ�ú�count
		//feature based parallel, will sync in tree.Split inner function
		void PerformSplit(RegressionTree& tree, int bestLeaf, Fvec& targets,
			int& LTEChild, int& GTChild)
		{
			const SplitInfo& bestSplitInfo = _bestSplitInfoPerLeaf[bestLeaf];
			int newInteriorNodeIndex = tree.Split(bestLeaf, bestSplitInfo.Feature, bestSplitInfo.Threshold, bestSplitInfo.LTEOutput, bestSplitInfo.GTOutput, bestSplitInfo.Gain, bestSplitInfo.GainPValue);
			GTChild = ~tree.GTChild(newInteriorNodeIndex);
			LTEChild = bestLeaf;
			Partitioning.Split(bestLeaf, TrainData.Features[bestSplitInfo.Feature].Bins, bestSplitInfo.Threshold, GTChild);
			if (rabit::GetWorldSize() > 1)
			{
				gezi::Notifer notifer("Broadcast partitioning");
				//Rabit::Broadcast(Partitioning, bestSplitInfo.Feature % rabit::GetWorldSize());
				int root = bestSplitInfo.Feature % rabit::GetWorldSize();
				rabit::Broadcast(&Partitioning.Documents(), root);
				rabit::Broadcast(&Partitioning.LeafBegin(), root);
				rabit::Broadcast(&Partitioning.LeafCount(), root);
			}
		}

		void SetBestFeatureForLeaf(LeafSplitCandidates& leafSplitCandidates, int bestFeature)
		{
			int leaf = leafSplitCandidates.LeafIndex;
			_bestSplitInfoPerLeaf[leaf] = leafSplitCandidates.FeatureSplitInfo[bestFeature];
			if (rabit::GetWorldSize() > 1)
			{
				gezi::Notifer notifer("Allreduce for best split info", 2);
				ufo::Allreduce<SplitInfo, SplitInfo::Reduce>(&_bestSplitInfoPerLeaf[leaf], 1);
			}
		}

		Float CalculateSplittedLeafOutput(int totalCount, Float sumTargets, Float sumWeights)
		{
			if (!HasWeights())
			{
				return (sumTargets / ((Float)totalCount));
			}
			if (_bsrMaxTreeOutput < 0.0)
			{
				return (sumTargets / sumWeights);
			}
			return (sumTargets / (2.0 * sumWeights));
		}

		//@TODO TLC ����splitFraction ���õ��Ǻ��� ��������ӿ��ٶ� �Ƿ��������featureFraction����ǰ������ˣ� ����ÿ�η���ɸѡ�������Ǳ���
		//���м�֦ ��һ���޷������� ���ٿ��� ��ô���ֱ��ǰ���˴��� ���������ܿ����������
		//@TODO FindBestFeatureFromGains(IEnumerable<Float> gains)
		int GetBestSplit(vector<SplitInfo>& featureSplitInfo)
		{
			if (_splitFraction == 1 || _preSplitCheck) //@TODO float problem?
			{
				return 	max_element(featureSplitInfo.begin(), featureSplitInfo.end(),
					[](const SplitInfo& l, const SplitInfo& r) {return l.Gain < r.Gain; }) - featureSplitInfo.begin();
			}
			else
			{
				return max_pos_rand(featureSplitInfo, _rand, _splitFraction, [](const SplitInfo& l, const SplitInfo& r) {return l.Gain < r.Gain; });
			}
		}

		int GetBestSplit(vector<SplitInfo*>& featureSplitInfo)
		{
			if (_splitFraction == 1 || _preSplitCheck)
			{
				return 	max_element(featureSplitInfo.begin(), featureSplitInfo.end(),
					[](const SplitInfo* const l, const SplitInfo* const r) {return l->Gain < r->Gain; }) - featureSplitInfo.begin();
			}
			else
			{
				return max_pos_rand(featureSplitInfo, _rand, _splitFraction, [](const SplitInfo* const l, const SplitInfo* const r) {return l->Gain < r->Gain; });
			}
		}

		void FindAndSetBestFeatureForLeaf(LeafSplitCandidates& leafSplitCandidates)
		{
			int bestFeature = GetBestSplit(leafSplitCandidates.FeatureSplitInfo);
			SetBestFeatureForLeaf(leafSplitCandidates, bestFeature);
		}

		Fvec _tempTargetVec;
		//@TODO �ܱ�Ť 
		Fvec& GetTargetWeights()
		{
			if (TargetWeights)
			{
				return *(TargetWeights);
			}
			return _tempTargetVec;
		}

		Float GetLeafSplitGain(int count, Float starget, Float sweight)
		{
			if (!HasWeights())
			{
				return ((starget * starget) / ((Float)count));
			}
			Float astarget = std::abs(starget);
			if ((_bsrMaxTreeOutput > 0.0) && (astarget >= ((2.0 * sweight) * _bsrMaxTreeOutput)))
			{
				return ((4.0 * _bsrMaxTreeOutput) * (astarget - (_bsrMaxTreeOutput * sweight)));
			}
			return ((starget * starget) / sweight);
		}

		void FindBestSplitOfRoot(Fvec& targets)
		{
			if (Partitioning.NumDocs() == TrainData.NumDocs)
			{ //��ǰ�������֧��ͳ���ܵ�target֮��,_filterZeros = false
				_smallerChildSplitCandidates.Initialize(targets, GetTargetWeights(), _filterZeros);
			} //��Ӧ����ӿ� _smallerChildSplitCandidates ��Ӧ��LeafIndex = 0
			else
			{
				_smallerChildSplitCandidates.Initialize(0, Partitioning, targets, GetTargetWeights(), _filterZeros);
			}
			_parentHistogramArray = NULL;
			_histogramArrayPool.Get(0, _smallerChildHistogramArray); //��pool�г�ȡһ��histogramλ��
			_largerChildSplitCandidates.Initialize(); //larger clearҲ���ǲ����� LeafIndex =-1

#pragma omp parallel for
			for (int featureIndex = 0; featureIndex < TrainData.NumFeatures; featureIndex++)
			{
				if (IsFeatureOk(featureIndex))
				{
					FindBestThresholdForFeature(featureIndex);
				}
			}
			FindAndSetBestFeatureForLeaf(_smallerChildSplitCandidates);
		}

		vector<Float> GetGains(const vector<SplitInfo>& infos)
		{
			return from(infos)
				>> select([](const SplitInfo& a) { return a.Gain; })
				>> to_vector();
		}

		//���ٶ����ķ���best split���ڶԱ���֤����ʵ�ֵ���ȷ��
		void FindBestSplitOfSiblingsSimple(int LTEChild, int GTChild, DocumentPartitioning& partitioning, const Fvec& targets)
		{
			int numDocsInLTEChild = partitioning.NumDocsInLeaf(LTEChild);
			int numDocsInGTChild = partitioning.NumDocsInLeaf(GTChild);
			if ((numDocsInGTChild < (_minDocsInLeaf * 2)) && (numDocsInLTEChild < (_minDocsInLeaf * 2)))
			{
				_bestSplitInfoPerLeaf[LTEChild].Gain = -std::numeric_limits<Float>::infinity();
				_bestSplitInfoPerLeaf[GTChild].Gain = -std::numeric_limits<Float>::infinity();
			}
			else
			{
				_parentHistogramArray = NULL;
				/*	if (numDocsInLTEChild >= numDocsInGTChild)
					{
					std::swap(LTEChild, GTChild);
					}*/ //for debug the same sequece as no simple implemention
				_smallerChildSplitCandidates.Initialize(LTEChild, partitioning, targets, GetTargetWeights(), _filterZeros);
				_largerChildSplitCandidates.Initialize(GTChild, partitioning, targets, GetTargetWeights(), _filterZeros);
				_histogramArrayPool.SimpleGet(LTEChild, _smallerChildHistogramArray);
				_histogramArrayPool.SimpleGet(GTChild, _largerChildHistogramArray);
			}
			{
#pragma omp parallel for
				for (int featureIndex = 0; featureIndex < TrainData.NumFeatures; featureIndex++)
				{
					if (IsFeatureOk(featureIndex))
					{
						FindBestThresholdForFeatureSimple(featureIndex);
					}
				}
			}
			FindAndSetBestFeatureForLeaf(_smallerChildSplitCandidates);
			FindAndSetBestFeatureForLeaf(_largerChildSplitCandidates);
		}

		void FindBestSplitOfSiblings(int LTEChild, int GTChild, DocumentPartitioning& partitioning, const Fvec& targets)
		{
			//AutoTimer timer("FindBestSplitOfSiblings");
			int numDocsInLTEChild = partitioning.NumDocsInLeaf(LTEChild);
			int numDocsInGTChild = partitioning.NumDocsInLeaf(GTChild);
			if ((numDocsInGTChild < (_minDocsInLeaf * 2)) && (numDocsInLTEChild < (_minDocsInLeaf * 2)))
			{//����Ҷ�Ӷ��޷��ٷ��� ����Υ��Ҷ������Сinstance��Ŀ����
				/*	_bestSplitInfoPerLeaf[LTEChild]->Gain = -std::numeric_limits<Float>::infinity();
				_bestSplitInfoPerLeaf[GTChild]->Gain = -std::numeric_limits<Float>::infinity();*/
				_bestSplitInfoPerLeaf[LTEChild].Gain = -std::numeric_limits<Float>::infinity();
				_bestSplitInfoPerLeaf[GTChild].Gain = -std::numeric_limits<Float>::infinity();
			}
			else
			{
				_parentHistogramArray = NULL;
				if (numDocsInLTEChild < numDocsInGTChild)
				{
					_smallerChildSplitCandidates.Initialize(LTEChild, partitioning, targets, GetTargetWeights(), _filterZeros);
					_largerChildSplitCandidates.Initialize(GTChild, partitioning, targets, GetTargetWeights(), _filterZeros);
					//���ݷ��ѵĹ���LTEChild��ǰ�Ǹ��ڵ�����(����Ҷ��0���� ��ô0���ڲ��ڵ�����ڲ����ڵ㣬����0��ΪҶ�ӽڵ㣬����һ���µ�Ҷ�ӽڵ���ԭ�����Ҷ�ӽڵ���+1)
					if (_histogramArrayPool.Get(LTEChild, _largerChildHistogramArray))
					{
						_parentHistogramArray = _largerChildHistogramArray;
					}
					_histogramArrayPool.Steal(LTEChild, GTChild); // GTChild -> _largerChildHistogramArray
					_histogramArrayPool.Get(LTEChild, _smallerChildHistogramArray); // LTEChild -> _smallerChildHistogramArray
				}
				else
				{
					_smallerChildSplitCandidates.Initialize(GTChild, partitioning, targets, GetTargetWeights(), _filterZeros);
					_largerChildSplitCandidates.Initialize(LTEChild, partitioning, targets, GetTargetWeights(), _filterZeros);
					if (_histogramArrayPool.Get(LTEChild, _largerChildHistogramArray))
					{
						_parentHistogramArray = _largerChildHistogramArray;
					}
					_histogramArrayPool.Get(GTChild, _smallerChildHistogramArray);// LTEChild -> _largerChildHistogramArray, GTChild -> _smallerChildHistogramArray
					//��������������Ǽ���_smallerChildHistogramArray����Instance�ĺ�������
				}
				{
					//AutoTimer timer("FindBestThresholdForFeature");
#pragma omp parallel for
					for (int featureIndex = 0; featureIndex < TrainData.NumFeatures; featureIndex++)
					{
						if (IsFeatureOk(featureIndex))
						{
							FindBestThresholdForFeature(featureIndex);
						}
					}
				}
				FindAndSetBestFeatureForLeaf(_smallerChildSplitCandidates);
				FindAndSetBestFeatureForLeaf(_largerChildSplitCandidates);
			}
		}

		void FindBestThresholdForFeatureSimple(int featureIndex)
		{
			(*_smallerChildHistogramArray)[featureIndex].SumupWeighted(featureIndex, _smallerChildSplitCandidates.NumDocsInLeaf, _smallerChildSplitCandidates.SumTargets, _smallerChildSplitCandidates.SumWeights, _smallerChildSplitCandidates.Targets, _smallerChildSplitCandidates.Weights, _smallerChildSplitCandidates.DocIndices);
			FindBestThresholdFromHistogram((*_smallerChildHistogramArray)[featureIndex], _smallerChildSplitCandidates, featureIndex);
			(*_largerChildHistogramArray)[featureIndex].SumupWeighted(featureIndex, _largerChildSplitCandidates.NumDocsInLeaf, _largerChildSplitCandidates.SumTargets, _largerChildSplitCandidates.SumWeights, _largerChildSplitCandidates.Targets, _largerChildSplitCandidates.Weights, _largerChildSplitCandidates.DocIndices);
			FindBestThresholdFromHistogram((*_largerChildHistogramArray)[featureIndex], _largerChildSplitCandidates, featureIndex);
		}
		//��������������Ӧ�� ��ǰҶ���ڲ���ֱ��ͼͳ��,Ȼ�����ֱ��ͼͳ���ҵ���ѷ�����ֵ
		void FindBestThresholdForFeature(int featureIndex)
		{
			//AutoTimer("FindBestThresholdForFeature");
			if (_parentHistogramArray && !(*_parentHistogramArray)[featureIndex].IsSplittable)
			{
				//VLOG(4) << "set smaller is splittable false";
				(*_smallerChildHistogramArray)[featureIndex].IsSplittable = false;
			}
			else
			{
				//VLOG(4) << "_smaller sumupweighted" << _smallerChildSplitCandidates.DocIndices.size();
				//--Histogramͳ�����������Ӧ_smallerChildSplitCandidates������ܷ�Ͱֱ��ͼ��Ϣ
				(*_smallerChildHistogramArray)[featureIndex].SumupWeighted(featureIndex, _smallerChildSplitCandidates.NumDocsInLeaf, _smallerChildSplitCandidates.SumTargets, _smallerChildSplitCandidates.SumWeights, _smallerChildSplitCandidates.Targets, _smallerChildSplitCandidates.Weights, _smallerChildSplitCandidates.DocIndices);
				//--���Ҷ�Ӧ_smallerChildSplitCandidates����,��Ӧ��featureIndex,�������ѵ�(Ͱ��Ŀ����)�ķ�������,������_smallerChildSplitCandidates�б���
				//��featureIndex����ѷ�����Ϣ
				FindBestThresholdFromHistogram((*_smallerChildHistogramArray)[featureIndex], _smallerChildSplitCandidates, featureIndex);
				if (_largerChildSplitCandidates.LeafIndex >= 0) //FindBestSplitOfRoot��ʱ��������-1,������
				{
					//or affine tree
					if (!_parentHistogramArray)
					{ //���pool�㹻�� ���������� ����˵parent Histogram��Ϣû��cache�� ��ô���¼���
						//VLOG(0) << "_parentHistogramArray null�� larger child set " << _largerChildSplitCandidates.DocIndices.size();
						(*_largerChildHistogramArray)[featureIndex].SumupWeighted(featureIndex, _largerChildSplitCandidates.NumDocsInLeaf, _largerChildSplitCandidates.SumTargets, _largerChildSplitCandidates.SumWeights, _largerChildSplitCandidates.Targets, _largerChildSplitCandidates.Weights, _largerChildSplitCandidates.DocIndices);
					}
					else
					{ //�Ż�����ͨ������ֱ�ӵõ��ڵ�϶��Ҷ�ӵ�Histogram��Ϣ
						//VLOG(4) << "_parentHistogramArray not null substract";
						(*_largerChildHistogramArray)[featureIndex].Subtract((*_smallerChildHistogramArray)[featureIndex]);
					}
					//VLOG(4) << "Find from _lager histogram" << _largerChildSplitCandidates.DocIndices.size();
					FindBestThresholdFromHistogram((*_largerChildHistogramArray)[featureIndex], _largerChildSplitCandidates, featureIndex);
				}
			}
		}

		//��Ե�ǰҶ�ӽڵ��ֱ��ͼͳ��,ʹ�� LeafSplitCandidates������doc��ͳ�ƣ����㵱ǰҶ��
		//��Ӧ��ǰ��������ѵķ�����Ϣ
		void FindBestThresholdFromHistogram(FeatureHistogram& histogram, LeafSplitCandidates& leafSplitCandidates, int feature)
		{
			//AutoTimer timer("FindBestThresholdFromHistogram");
			Float bestSumLTETargets = std::numeric_limits<Float>::quiet_NaN();
			Float bestSumLTEWeights = std::numeric_limits<Float>::quiet_NaN();
			Float bestShiftedGain = -std::numeric_limits<Float>::infinity();
			Float trust = TrainData.Features[feature].Trust;
			int bestLTECount = -1;
			uint bestThreshold = (uint)histogram.NumFeatureValues;
			Float eps = 1E-10;
			Float sumLTETargets = 0.0;
			Float sumLTEWeights = eps;
			int LTECount = 0;
			int totalCount = leafSplitCandidates.NumDocsInLeaf;
			Float sumTargets = leafSplitCandidates.SumTargets;
			Float sumWeights = leafSplitCandidates.SumWeights + (2.0 * eps);
			Float gainShift = GetLeafSplitGain(totalCount, sumTargets, sumWeights);
			Float minShiftedGain = (_gainConfidenceInSquaredStandardDeviations <= 0.0) ? 0.0 :
				((((_gainConfidenceInSquaredStandardDeviations * leafSplitCandidates.VarianceTargets()) * totalCount) / ((Float)(totalCount - 1))) + gainShift);
			histogram.IsSplittable = false;
			Float minDocsForThis = ((Float)_minDocsInLeaf) / trust;
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
					Float sumGTTargets = sumTargets - sumLTETargets;
					Float sumGTWeights = sumWeights - sumLTEWeights;
					Float currentShiftedGain = GetLeafSplitGain(LTECount, sumLTETargets, sumLTEWeights) + GetLeafSplitGain(GTCount, sumGTTargets, sumGTWeights);
					if (currentShiftedGain >= minShiftedGain)
					{
						histogram.IsSplittable = true;
						if (_entropyCoefficient > 0.0)
						{
							Float entropyGain = ((totalCount * std::log((Float)totalCount)) - (LTECount * std::log((Float)LTECount))) - (GTCount * std::log((Float)GTCount));
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
			Float usePenalty = (_featureUseCount[feature] == 0) ? _featureFirstUsePenalty : (_featureReusePenalty * std::log((Float)(_featureUseCount[feature] + 1)));
			leafSplitCandidates.FeatureSplitInfo[feature].Gain = ((bestShiftedGain - gainShift) * trust) - usePenalty;
			Float erfcArg = std::sqrt(((bestShiftedGain - gainShift) * (totalCount - 1)) / ((2.0 * leafSplitCandidates.VarianceTargets()) * totalCount));
			leafSplitCandidates.FeatureSplitInfo[feature].GainPValue = ProbabilityFunctions::Erfc(erfcArg);
			PVAL2(leafSplitCandidates.FeatureSplitInfo[feature].LTEOutput, leafSplitCandidates.FeatureSplitInfo[feature].GTOutput);
			PVAL5(bestShiftedGain, gainShift, leafSplitCandidates.FeatureSplitInfo[feature].Gain, trust, usePenalty);
		}

		void SetRootModel(RegressionTree& tree, Fvec& targets)
		{
		}
	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of LEAST_SQUARES_REGRESSION_TREE_LEARNER_H_
