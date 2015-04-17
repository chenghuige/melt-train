/**
 *  ==============================================================================
 *
 *          \file   Trainers/FastRank/FastRank.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-07 16:09:21.545812
 *
 *  \Description:  对应TrainingApplicationBase
 *  ==============================================================================
 */

#ifndef TRAINERS__FAST_RANK__FAST_RANK_H_
#define TRAINERS__FAST_RANK__FAST_RANK_H_
#include "common_def.h"
#include "InstancesToDataset.h"
#include "MLCore/Trainer.h"
#include "Dataset.h"
#include "Ensemble.h"
#include "OptimizationAlgorithm.h"
#include "FastRankArguments.h"
#include "IGradientAdjuster.h"
#include "GradientDescent.h"
#include "TrivialGradientWrapper.h"
#include "BestStepRegressionGradientWrapper.h"
#include "QueryWeightsGradientWrapper.h"
#include "QueryWeightsBestResressionStepGradientWrapper.h"
#include "LeastSquaresRegressionTreeLearner.h"
#include "BaggingProvider.h"
#include "Predictors/FastRankPredictor.h"
#include "Prediction/Calibrate/CalibratorFactory.h"
namespace gezi {

	class FastRank : public Trainer
	{
	public:
		Dataset TrainSet;
		vector<Dataset> TestSets;
		Dataset ValidSet;

		Fmat InitTestScores;
		Fvec InitTrainScores;
		Fvec InitValidScores;
	public:
		FastRank()
		{
			//ParseArgs(); //这里就不行了 所以放置虚函数在构造函数有风险 pure virtual method called  因为ParseArgs里又调用了虚函数 	_args = GetArguments();
			//也就是说尽量不要再构造函数有虚函数 如果必须 只放置在最底层的derived class构造函数中
		}

		virtual void ShowHelp() override
		{
			fmt::print_line("DECLARE_bool(calibrate);");
			fmt::print_line("DECLARE_string(calibrator);");
			fmt::print_line("DECLARE_uint64(rs);");
			fmt::print_line("DECLARE_int32(iter);");
			fmt::print_line("DEFINE_int32(ntree, 100, \"numTrees: Number of trees/iteraiton number\");");
			fmt::print_line("DECLARE_double(lr);");
			fmt::print_line("DEFINE_int32(nl, 20, \"numLeaves: Number of leaves maximam allowed in each regression tree\");");
			fmt::print_line("DEFINE_int32(mil, 10, \"minInstancesInLeaf: Minimal instances in leaves allowd\");");
			fmt::print_line("DEFINE_bool(bsr, false, \"bestStepRankingRegressionTrees: \");");
			fmt::print_line("DEFINE_double(sp, 0.1, \"Sparsity level needed to use sparse feature representation, if 0.3 means be sparsify only if real data less then 30%, 0-1 the smaller more dense and faster but use more memeory\");");
			fmt::print_line("DEFINE_double(ff, 1, \"The fraction of features (chosen randomly) to use on each iteration\");");
			fmt::print_line("DEFINE_double(sf, 1, \"The fraction of features(chosen randomly) to use on each split\");");
			fmt::print_line("DEFINE_int32(mb, 255, \"Maximum number of distinct values (bins) per feature\");");

			fmt::print_line("int numTrees = 100;");
			fmt::print_line("int numLeaves = 20;");
			fmt::print_line("int minInstancesInLeaf = 10;");
			fmt::print_line("Float learningRate = 0.2;");
			fmt::print_line("bool calibrateOutput = true; //calibrate| use calibrator to gen probability?");
			fmt::print_line("string calibratorName = \"sigmoid\"; //calibrator| sigmoid/platt naive pav @TODO 移动到Trainer父类处理");
			fmt::print_line("int maxBins = 255; //mb|Maximum number of distinct values (bins) per feature");
			fmt::print_line("Float sparsifyRatio = 0.3;//sr|if not big data (large instances num, large feature num can set 0 so to be all dense) that will make running faster");
			fmt::print_line("unsigned randSeed = 0x7b; //rs|controls wether the expermient can reproduce, 0 means not reproduce rngSeed");
			fmt::print_line("bool randomStart = false; //rst|Training starts from random ordering (determined by /r1)");
			fmt::print_line("int histogramPoolSize = -1; //|[2, numLeaves - 1]");
			fmt::print_line("int maxTreeOutput = 100; //mo|Upper bound on absolute value of single tree output");
			fmt::print_line("bool affineRegressionTrees = false; //art|Use affine regression trees");
			fmt::print_line("bool allowDummyRootSplits = true; //dummies|When a root split is impossible, construct a dummy empty tree rather than fail");
			fmt::print_line("Float baggingTrainFraction = 0.7;//bagfrac|Percentage of training queries used in each bag");
			fmt::print_line("bool bestStepRankingRegressionTrees = false; //bsr|Use best ranking step trees");
			fmt::print_line("Float entropyCoefficient = 0; //e|The entropy (regularization) coefficient between 0 and 1");
			fmt::print_line("int derivativesSampleRate = 1; //dsr|same each query 1 in k times in the GetDerivatives function");
			fmt::print_line("Float smoothing = 0.0; //s|Smoothing paramter for tree regularization");
			fmt::print_line("bool compressEnsemble = false; //cmp|Compress the tree Ensemble");
			fmt::print_line("Float featureFirstUsePenalty = 0;");
			fmt::print_line("Float featureReusePenalty = 0;");
			fmt::print_line("Float softmaxTemperature = 0;");
			fmt::print_line("Float featureFraction = 1; //");
			fmt::print_line("Float splitFraction = 1; //");
			fmt::print_line("bool filterZeroLambdas = false;");
			fmt::print_line("Float gainConfidenceLevel = 0;");
		}
		/*	enum class OptimizationAlgorithm
			{
			GradientDescent,
			AcceleratedGradientDescent,
			ConjugateGradientDescent
			}*/

		virtual string GetParam() override
		{
			stringstream ss;
			ss << "numTrees:" << _args->numTrees << " "
				<< "numLeaves:" << _args->numLeaves << " "
				<< "minInstancesInLeaf:" << _args->minInstancesInLeaf << " "
				<< "learningRate:" << _args->learningRate << " "
				<< "featureFraction:" << _args->featureFraction;
			return ss.str();
		}

		//@TODO 
		virtual void ParseArgs() override;
		/*{
			_args = GetArguments();
			if (_args->histogramPoolSize < 2)
			{
			_args->histogramPoolSize = (_args->numLeaves * 2) / 3;
			}
			if (_args->histogramPoolSize >(_args->numLeaves - 1))
			{
			_args->histogramPoolSize = _args->numLeaves - 1;
			}
			}*/

		virtual void CustomizedTrainingIteration()
		{

		}

		void ConvertData(Instances& instances)
		{
			TrainSet = InstancesToDataset::Convert(instances, _args->maxBins, _args->sparsifyRatio);
		}

		BitArray* GetActiveFeatures(BitArray& activeFeatures)
		{
			BitArray* pactiveFeatures;
			if (_args->featureFraction == 1)
			{
				pactiveFeatures = &_activeFeatures;
			}
			else
			{
				activeFeatures = _activeFeatures;
				for (size_t i = 0; i < activeFeatures.size(); i++)
				{
					if (activeFeatures[i])
					{
						if (_rand->NextDouble() > _args->featureFraction)
						{
							activeFeatures[i] = false;
						}
					}
				}
				pactiveFeatures = &activeFeatures;
			}
			return pactiveFeatures;
		}

		void TrainCore()
		{
			int numTotalTrees = _args->numTrees;
			bool revertRandomStart = false;
			if (_args->randomStart && (_ensemble.NumTrees() < numTotalTrees))
			{
				VLOG(1) << "Randomizing start point";
				(_optimizationAlgorithm->TrainingScores)->RandomizeScores(_args->randSeed, false);
				revertRandomStart = true;
			}
			ProgressBar pb(numTotalTrees, "Ensemble trainning");
		
			BaggingProvider baggingProvider(TrainSet, _args->randSeed, _args->numLeaves, _args->baggingTrainFraction);
			while (_ensemble.NumTrees() < numTotalTrees)
			{
				DocumentPartitioning currentOutOfBagPartition;
				if (_args->baggingSize != 0 && _ensemble.NumTrees() % _args->baggingSize == 0)
				{
					baggingProvider.GenPartion(_optimizationAlgorithm->TreeLearner->Partitioning, currentOutOfBagPartition);
				}
				++pb;
				PVAL(_ensemble.NumTrees());
				BitArray activeFeatures;
				BitArray* pactiveFeatures = GetActiveFeatures(activeFeatures);
				_optimizationAlgorithm->TrainingIteration(*pactiveFeatures);
				if (_args->baggingSize > 0)
				{
					_ensemble.LastTree().AddOutputsToScores(_optimizationAlgorithm->TrainingScores->Dataset, 
						_optimizationAlgorithm->TrainingScores->Scores, 
						currentOutOfBagPartition.Documents());
				}
				CustomizedTrainingIteration();
				if (revertRandomStart)
				{
					revertRandomStart = false;
					VLOG(1) << "Reverting random score assignment";
					(_optimizationAlgorithm->TrainingScores)->RandomizeScores(_args->randSeed, true);
				}
				if (VLOG_IS_ON(3))
				{
					_ensemble.Back().Print(TrainSet.Features);
				}
			}
			_optimizationAlgorithm->FinalizeLearning(GetBestIteration());
		}

		void FeatureGainPrint(int level = 1)
		{
			VLOG(level) << "Per_feature_gain:\n" <<
				_ensemble.ToGainSummary(TrainSet.Features);
			VLOG(level) << "Per_feature_gain_end";
		}


		virtual void Train(Instances& instances) override
		{
			InnerTrain(instances);
			Finalize(instances);
		}

		virtual void InnerTrain(Instances& instances) override
		{
			ParseArgs();
			ConvertData(instances);
			Initialize();
			TrainCore();
			FeatureGainPrint();
		}

		virtual OptimizationAlgorithmPtr ConstructOptimizationAlgorithm()
		{
			_optimizationAlgorithm = make_shared<GradientDescent>(_ensemble, TrainSet, InitTrainScores, MakeGradientWrapper());
			_optimizationAlgorithm->Initialize(TrainSet, InitTrainScores);
			_optimizationAlgorithm->TreeLearner = ConstructTreeLearner();
			_optimizationAlgorithm->ObjectiveFunction = ConstructObjFunc();
			_optimizationAlgorithm->Smoothing = _args->smoothing;
			return _optimizationAlgorithm;
		}

	protected:
		virtual ObjectiveFunctionPtr ConstructObjFunc() = 0;
		virtual void InitializeTests() = 0;

		virtual TreeLearnerPtr ConstructTreeLearner()
		{
			PVAL(AreTargetsWeighted());
			return make_shared<LeastSquaresRegressionTreeLearner>(TrainSet, _args->numLeaves, _args->minInstancesInLeaf, _args->entropyCoefficient, 
				_args->featureFirstUsePenalty, _args->featureReusePenalty, _args->softmaxTemperature, _args->histogramPoolSize,
				_args->randSeed, _args->splitFraction, _args->preSplitCheck,
				_args->filterZeroLambdas, _args->allowDummyRootSplits, _args->gainConfidenceLevel,
				AreTargetsWeighted(), BsrMaxTreeOutput());
		}

		bool AreTrainWeightsUsed()
		{//只要是有weight数据 就使用 如果不使用 在melt框架部分确保weight数据为空即可
			//可行的方法是  将weight列视作attr 比如 -weight 3 可以 -attr 3忽略掉即可
			return true;
		}

		bool AreSamplesWeighted()
		{
			return (AreTrainWeightsUsed() && (!TrainSet.SampleWeights.empty()));
		}

		bool AreTargetsWeighted()
		{
			if (!AreSamplesWeighted())
			{
				return _args->bestStepRankingRegressionTrees;
			}
			return true;
		}

		Float BsrMaxTreeOutput()
		{
			if (_args->bestStepRankingRegressionTrees)
			{
				return _args->maxTreeOutput;
			}
			return -1.0;
		}

		virtual IGradientAdjusterPtr MakeGradientWrapper()
		{
			if (AreSamplesWeighted())
			{
				if (_args->bestStepRankingRegressionTrees)
				{
					return make_shared<QueryWeightsBestStepRegressionGradientWrapper>();
				}
				return make_shared<QueryWeightsGradientWrapper>();
			}
			return make_shared<TrivialGradientWrapper>();
		}

		virtual int GetBestIteration()
		{
			int bestIteration = _ensemble.NumTrees();
			//@TODO
			/*if (!cmd.writeLastEnsemble && (EarlyStoppingTest != null))
			{
			bestIteration = EarlyStoppingTest.BestIteration;
			}*/
			return bestIteration;
		}

		virtual void Initialize()
		{
			_rand = make_shared<Random>(random_engine(_args->randSeed));

			_activeFeatures.resize(TrainSet.Features.size(), true);
			for (size_t i = 0; i < TrainSet.Features.size(); i++)
			{
				if (TrainSet.Features[i].NumBins() <= 1)
				{
					_activeFeatures[i] = false;
				}
			}

			PrepareLabels();
			_optimizationAlgorithm = ConstructOptimizationAlgorithm();
			InitializeTests();
		}

		virtual void PrepareLabels() = 0;

		Fvec& GetInitScores(Dataset& set)
		{
			if (&set == &TrainSet)
			{
				return InitTrainScores;
			}
			if (&set == &ValidSet)
			{
				return InitValidScores;
			}
			for (size_t i = 0; (!TestSets.empty()) && (i < TestSets.size()); i++)
			{
				if (&set == &TestSets[i])
				{
					if (InitTestScores.empty())
					{
						return _tempScores;
					}
					return InitTestScores[i];
				}
			}
			throw new Exception("Queried for unknown set");
		}

		Fvec ComputeScoresSlow(Dataset& set)
		{
			Fvec scores(set.NumDocs);
			_ensemble.GetOutputs(set, scores);
			Fvec& initScores = GetInitScores(set);
			if (!initScores.empty())
			{
				if (scores.size() != initScores.size())
				{
					THROW("Length of initscores and scores mismatch");
				}
				for (size_t i = 0; i < scores.size(); i++)
				{
					scores[i] += initScores[i];
				}
			}
			return scores;
		}

		Fvec ComputeScoresSmart(Dataset& set)
		{
			if (!_args->compressEnsemble)
			{
				for (ScoreTrackerPtr st : _optimizationAlgorithm->TrackedScores)
				{
					if (&(st->Dataset) == &set)
					{
						Fvec result = move(st->Scores);
						return result;
					}
				}
			}
			return ComputeScoresSlow(set);
		}

		virtual FastRankArgumentsPtr GetArguments() = 0;
	protected:
		FastRankArgumentsPtr _args;

		Ensemble _ensemble;
		OptimizationAlgorithmPtr _optimizationAlgorithm = nullptr;

		Fvec _tempScores;

		BitArray _activeFeatures;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__FAST_RANK_H_
