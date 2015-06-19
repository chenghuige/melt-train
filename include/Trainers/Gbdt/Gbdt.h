/**
 *  ==============================================================================
 *
 *          \file   Trainers/Gbdt/Gbdt.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-07 16:09:21.545812
 *
 *  \Description:  对应TrainingApplicationBase
 *  ==============================================================================
 */

#ifndef TRAINERS__GBDT__GBDT_H_
#define TRAINERS__GBDT__GBDT_H_
#include "common_def.h"
#include "InstancesToDataset.h"
#include "MLCore/Trainer.h"
#include "Dataset.h"
#include "Ensemble.h"
#include "OptimizationAlgorithm.h"
#include "GbdtArguments.h"
#include "IGradientAdjuster.h"
#include "GradientDescent.h"
#include "TrivialGradientWrapper.h"
#include "BestStepRegressionGradientWrapper.h"
#include "QueryWeightsGradientWrapper.h"
#include "QueryWeightsBestResressionStepGradientWrapper.h"
#include "LeastSquaresRegressionTreeLearner.h"
#include "BaggingProvider.h"
#include "Predictors/GbdtPredictor.h"
#include "Prediction/Calibrate/CalibratorFactory.h"
#include "Prediction/Instances/instances_util.h"
#include "rabit_util.h"
namespace gezi {

	class Gbdt : public ValidatingTrainer
	{
	public:
		Dataset TrainSet;
		Instances* InputInstances = NULL;
	public:
		Gbdt()
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

		virtual PredictorPtr CreatePredictor() override
		{
			vector<Float> featureGainVec = _ensemble.ToGainVec(TrainSet.Features); //must before CreatePredictor for trees will move to predictor instead
			vector<OnlineRegressionTree> trees = _ensemble.GetOnlineRegressionTrees();
			auto predictor = CreatePredictor(trees);
			predictor->SetFeatureNames(InputInstances->FeatureNames());
			predictor->SetFeatureGainVec(move(featureGainVec));
			return predictor;
		}

		virtual PredictorPtr CreatePredictor(vector<OnlineRegressionTree>& trees) = 0;

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

		void ReInitialize(int step)
		{
			_args->randSeed += step * 1024;
			//_rand = make_shared<Random>(random_engine(_args->randSeed));
			_optimizationAlgorithm = ConstructOptimizationAlgorithm();
		}

		//step means 第几个bagging
		void TrainCore(int step = 1)
		{
			int numTotalTrees = step * _args->numTrees;
			bool revertRandomStart = false;
			if (_args->randomStart && (_ensemble.NumTrees() < numTotalTrees))
			{
				VLOG(1) << "Randomizing start point";
				(_optimizationAlgorithm->TrainingScores)->RandomizeScores(_args->randSeed, false);
				revertRandomStart = true;
			}
			ProgressBar pb(numTotalTrees, "Ensemble trainning");
			BaggingProvider baggingProvider(TrainSet, _args->randSeed, _args->numLeaves, _args->baggingTrainFraction);
			DocumentPartitioning currentOutOfBagPartition;
			while (_ensemble.NumTrees() < numTotalTrees)
			{
				//if (_args->numBags == 1 && _args->baggingSize != 0 && _ensemble.NumTrees() % _args->baggingSize == 0)
				if (_args->baggingSize != 0 && _ensemble.NumTrees() % _args->baggingSize == 0)
				{
					baggingProvider.GenPartion(_optimizationAlgorithm->TreeLearner->Partitioning, currentOutOfBagPartition);
				}
				++pb;
				BitArray activeFeatures;
				BitArray* pactiveFeatures = GetActiveFeatures(activeFeatures);
				_optimizationAlgorithm->TrainingIteration(*pactiveFeatures);
				//if (_args->numBags == 1 && _args->baggingSize > 0)
				if (_args->baggingSize > 0)
				{
					_ensemble.LastTree().AddOutputsToScores(*_optimizationAlgorithm->TrainingScores->Dataset,
						_optimizationAlgorithm->TrainingScores->Scores,
						currentOutOfBagPartition.Documents());
				}
				CustomizedTrainingIteration();
				if (_validating && (_ensemble.NumTrees() % _evaluateFrequency == 0 || _ensemble.NumTrees() == numTotalTrees))
				{
					std::cerr << "Trees: " << _ensemble.NumTrees() << " Leaves: " << _ensemble.Back().NumLeaves << " \n";
					//if (_ensemble.NumTrees() == 197)
					//{
					//	FeatureNamesVector fnv(TrainSet.FeatureNames());
					//	_ensemble.Back().SetFeatureNames(fnv);
					//	//_ensemble.Back().SetFeatureNames(TrainSet.FeatureNames()); //注意错误！ 因为内部是指针 而临时变量析构了。。 野的指针了。。
					//	std::cerr << "Print Instances\n";
					//	_ensemble.Back().Print(_validationSets[0][1323]->features);
					//	std::cerr << "Print DataSet\n";
					//	_ensemble.Back().Print(TrainSet[1323]);
					//}
				}
				if (ValidatingTrainer::Evaluate(_ensemble.NumTrees(), _ensemble.NumTrees() == numTotalTrees))
				{
					break;
				}
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
			if (VLOG_IS_ON(level))
			{
				std::cerr << "Per_feature_gain" << std::endl;
				std::cerr << _ensemble.ToGainSummary(TrainSet.Features, _args->maxFeaturesShow);
				std::cerr << "Per_feature_gain_end";
			}
		}

		virtual void Train(Instances& instances) override
		{
			VLOG(5) << "Train in gbdt";
			InnerTrain(instances);
			Finalize(instances);
		}

		virtual void InnerTrain(Instances& instances) override
		{
			ParseArgs();
			InputInstances = &instances;
			if (_args->numBags == 1)
			{
				ConvertData(instances);
				Initialize();
				TrainCore();
			}
			else
			{
				Random rand(_args->randSeed);
				//如果每个内部只有一个树 随机森林 , @TODO可并行  目前公用一个TrainSet TrainScores等等还不能并行
				//当然也可以将bagging的支持移动到外围方便并行 统一ensemble框架即可 @TODO
				//使用rabit每个机器运行一个 @TODO
				for (int i = 1; i <= _args->numBags; i++)
				{//boostStrap貌似效果不好没什么用处
					Instances partionInstances = _args->boostStrap ?
						InstancesUtil::GenBootstrapInstances(instances, rand, _args->bootStrapFraction) :
						InstancesUtil::GenPartionInstances(instances, rand, _args->nbaggingTrainFraction);
					ValidatingTrainer::SetSelfEvaluateInstances(partionInstances);
					ConvertData(partionInstances); //modify TrainSet
					Initialize(i);
					ValidatingTrainer::SetScale((double)i);
					TrainCore(i);
				}
				for (int t = 0; t < _ensemble.NumTrees(); t++)
				{
					_ensemble.Tree(t).ScaleOutputsBy(1.0 / ((double)_args->numBags));
				}
			}
			FeatureGainPrint();
		}

		virtual OptimizationAlgorithmPtr ConstructOptimizationAlgorithm()
		{ //Scores.back() is TrainingScores
			//@TODO 这个构造函数里面的TrainScores其实没有用。。。
			_optimizationAlgorithm = make_shared<GradientDescent>(_ensemble, TrainSet, TrainScores, MakeGradientWrapper());
			//_optimizationAlgorithm->Initialize(TrainSet, Scores.back()); //move it after InitalizeTests hack!
			_optimizationAlgorithm->TreeLearner = ConstructTreeLearner();
			_optimizationAlgorithm->ObjectiveFunction = ConstructObjFunc();
			_optimizationAlgorithm->Smoothing = _args->smoothing;
			return _optimizationAlgorithm;
		}

		ScoreTrackerPtr ConstructScoreTracker(const Instances& set, int index)
		{ //如果bagging 后续再次调用 没有再生产新的ScoreTracer 而是在之前基础上继续
			return _optimizationAlgorithm->GetScoreTracker(format("tes[{}]", index), _validationSets[index], Scores[index]);
		}

	protected:
		void ConstructValidatingScoreTrackers()
		{
			for (size_t i = 0; i < _validationSets.size(); i++)
			{
				ConstructScoreTracker(_validationSets[i], i);
			}
		}
		virtual ObjectiveFunctionPtr ConstructObjFunc() = 0;
		virtual void InitializeTests()
		{
			//后面 _optimizationAlgorithm 会肯定添加一个Train的 ScoreTracker
			if (_validating)
			{
				ConstructValidatingScoreTrackers();
			}
		}

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

		//@why bestStepRankingRegressionTrees 具体什么意思 很难理解 AdjustOutput的时候不再使用 _weight平滑 但是LeastSquareRegression中使用了weight
		//default的策略恰恰相反
		virtual IGradientAdjusterPtr MakeGradientWrapper()
		{
			if (AreSamplesWeighted())
			{
				if (_args->bestStepRankingRegressionTrees)
				{
					return make_shared<QueryWeightsBestStepRegressionGradientWrapper>();
				}
				else
				{
					return make_shared<QueryWeightsGradientWrapper>();
				}
			}
			else
			{
				if (_args->bestStepRankingRegressionTrees)
				{
					return make_shared<BestStepRegressionGradientWrapper>();
				}
				else
				{
					return make_shared<TrivialGradientWrapper>();
				}
			}
		}

		virtual int GetBestIteration()
		{
			int bestIteration = _ensemble.NumTrees();
			if (_earlyStop && _useBestStage)
			{
				bestIteration = ValidatingTrainer::BestIteration();
				VLOG(0) << "Final tree num will be " << bestIteration;
			}
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
			//gezi::zeroset(TrainScores); //对应Bagging 一上来就值不对了 是因为采样的原因。。 原因在于bagfraction 如果是1 ok
			gezi::reset_vec(TrainScores, TrainSet.NumDocs, 0.0);
			_optimizationAlgorithm->Initialize(TrainSet, TrainScores);
		}

		void Initialize(int step)
		{
			if (step != 1)
			{
				_args->randSeed += step * 1024;
			}

			Initialize();
		}

		virtual void PrepareLabels() = 0;

		//Fvec& GetInitScores(Dataset& set)
		//{
		//	if (&set == &TrainSet)
		//	{
		//		return InitTrainScores;
		//	}
		//	//if (&set == &ValidSet)
		//	//{
		//	//	return InitValidScores;
		//	//}
		//	//for (size_t i = 0; (!TestSets.empty()) && (i < TestSets.size()); i++)
		//	//{
		//	//	if (&set == &TestSets[i])
		//	//	{
		//	//		if (InitTestScores.empty())
		//	//		{
		//	//			return _tempScores;
		//	//		}
		//	//		return InitTestScores[i];
		//	//	}
		//	//}
		//	THROW("Queried for unknown set");
		//}

		//Fvec ComputeScoresSlow(Dataset& set)
		//{
		//	EvaluateScores.resize(set.size());
		//	_ensemble.GetOutputs(set, scores);
		//	Fvec& initScores = GetInitScores(set);
		//	if (!initScores.empty())
		//	{
		//		if (EvaluateScores.size() != initScores.size())
		//		{
		//			THROW("Length of initscores and scores mismatch");
		//		}
		//		for (size_t i = 0; i < scores.size(); i++)
		//		{
		//			scores[i] += initScores[i];
		//		}
		//	}
		//	return scores;
		//}

		//Fvec ComputeScoresSmart(Dataset& set)
		//{
		//	if (!_args->compressEnsemble)
		//	{
		//		for (ScoreTrackerPtr st : _optimizationAlgorithm->TrackedScores)
		//		{
		//			if (st->Dataset == &set)
		//			{
		//				//Fvec result = move(st->Scores); //如果没有validating self 那么只调用一次 ComputeScoresSmart(用于clibrate)可以用move
		//				Fvec result = st->Scores;  
		//				return result;
		//			}
		//		}
		//	}
		//}

		//@FIXME for bagging with sampling fraction might better use orginal TrainSet/InoputInstances and their tracing score
		//如果没有validate 原始的Instance 可以再做一次计算？ ScoreTracker没有记录
		Fvec& GetTrainSetScores()
		{
			return TrainScores;
		}

		virtual GbdtArgumentsPtr CreateArguments() = 0;
	protected:
		GbdtArgumentsPtr _args;

		Ensemble _ensemble;
		OptimizationAlgorithmPtr _optimizationAlgorithm = nullptr;

		Fvec _tempScores;

		BitArray _activeFeatures;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__GBDT__GBDT_H_
