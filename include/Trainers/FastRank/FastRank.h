/**
 *  ==============================================================================
 *
 *          \file   Trainers/FastRank/FastRank.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-07 16:09:21.545812
 *
 *  \Description:
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
#include "LeastSquaresRegressionTreeLearner.h"
#include "Predictors/FastRankPredictor.h"
#include "Prediction/Calibrate/CalibratorFactory.h"
namespace gezi {

	class FastRank : public Trainer
	{
	public:
		Dataset TrainSet;
		vector<Dataset> TestSets;
		Dataset ValidSet;

		dmat InitTestScores;
		dvec InitTrainScores;
		dvec InitValidScores;

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
		virtual void ParseArgs();
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
			while (_ensemble.NumTrees() < numTotalTrees)
			{
				++pb;
				
				BitArray activeFeatures;
				BitArray* pactiveFeatures = GetActiveFeatures(activeFeatures);
			
				_optimizationAlgorithm->TrainingIteration(*pactiveFeatures);
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
			VLOG(level) << "Per feature gain:\n" <<
				_ensemble.ToGainSummary(TrainSet.Features);
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
			return make_shared<LeastSquaresRegressionTreeLearner>(TrainSet, _args->numLeaves, _args->minInstancesInLeaf, _args->entropyCoefficient, _args->featureFirstUsePenalty, _args->featureReusePenalty, _args->softmaxTemperature, _args->histogramPoolSize, _args->randSeed, _args->splitFraction, _args->filterZeroLambdas, _args->allowDummyRootSplits, _args->gainConfidenceLevel, AreTargetsWeighted(), BsrMaxTreeOutput());
		}

		bool AreTrainWeightsUsed()
		{
			return true;
		}

		bool AreSamplesWeighted()
		{ //@TODO weight选项不可用
			return (AreTrainWeightsUsed() && (!TrainSet.SampleWeights.empty()));
		}

		bool AreTargetsWeighted()
		{
			return false; //@TODO @FIXME bestStepRankingRegressionTrees选项不可用？
			//即使bestStepRankingRegressionTrees已经设置 结果和tlc仍然不一致
			if (!AreSamplesWeighted())
			{
				return _args->bestStepRankingRegressionTrees;
			}
			return true;
		}

		double BsrMaxTreeOutput()
		{
			if (_args->bestStepRankingRegressionTrees)
			{
				return _args->maxTreeOutput;
			}
			return -1.0;
		}

		virtual IGradientAdjusterPtr MakeGradientWrapper()
		{
			if (_args->bestStepRankingRegressionTrees)
			{
				return make_shared<BestStepRegressionGradientWrapper>();
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

		dvec& GetInitScores(Dataset& set)
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

		dvec ComputeScoresSlow(Dataset& set)
		{
			dvec scores(set.NumDocs);
			_ensemble.GetOutputs(set, scores);
			dvec& initScores = GetInitScores(set);
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

		dvec ComputeScoresSmart(Dataset& set)
		{
			if (!_args->compressEnsemble)
			{
				for (ScoreTrackerPtr st : _optimizationAlgorithm->TrackedScores)
				{
					if (&(st->Dataset) == &set)
					{
						dvec result = move(st->Scores);
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

		dvec _tempScores;

		BitArray _activeFeatures;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__FAST_RANK_H_
