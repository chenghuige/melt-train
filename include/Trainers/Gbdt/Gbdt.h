/**
 *  ==============================================================================
 *
 *          \file   Trainers/Gbdt/Gbdt.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-07 16:09:21.545812
 *
 *  \Description:  ��ӦTrainingApplicationBase
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
			//ParseArgs(); //����Ͳ����� ���Է����麯���ڹ��캯���з��� pure virtual method called  ��ΪParseArgs���ֵ������麯�� 	_args = GetArguments();
			//Ҳ����˵������Ҫ�ٹ��캯�����麯�� ������� ֻ��������ײ��derived class���캯����
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
			fmt::print_line("string calibratorName = \"sigmoid\"; //calibrator| sigmoid/platt naive pav @TODO �ƶ���Trainer���ദ��");
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

		//step means �ڼ���bagging
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
					//	//_ensemble.Back().SetFeatureNames(TrainSet.FeatureNames()); //ע����� ��Ϊ�ڲ���ָ�� ����ʱ���������ˡ��� Ұ��ָ���ˡ���
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
				//���ÿ���ڲ�ֻ��һ���� ���ɭ�� , @TODO�ɲ���  Ŀǰ����һ��TrainSet TrainScores�ȵȻ����ܲ���
				//��ȻҲ���Խ�bagging��֧���ƶ�����Χ���㲢�� ͳһensemble��ܼ��� @TODO
				//ʹ��rabitÿ����������һ�� @TODO
				for (int i = 1; i <= _args->numBags; i++)
				{//boostStrapò��Ч������ûʲô�ô�
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
			//@TODO ������캯�������TrainScores��ʵû���á�����
			_optimizationAlgorithm = make_shared<GradientDescent>(_ensemble, TrainSet, TrainScores, MakeGradientWrapper());
			//_optimizationAlgorithm->Initialize(TrainSet, Scores.back()); //move it after InitalizeTests hack!
			_optimizationAlgorithm->TreeLearner = ConstructTreeLearner();
			_optimizationAlgorithm->ObjectiveFunction = ConstructObjFunc();
			_optimizationAlgorithm->Smoothing = _args->smoothing;
			return _optimizationAlgorithm;
		}

		ScoreTrackerPtr ConstructScoreTracker(const Instances& set, int index)
		{ //���bagging �����ٴε��� û���������µ�ScoreTracer ������֮ǰ�����ϼ���
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
			//���� _optimizationAlgorithm ��϶����һ��Train�� ScoreTracker
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
		{//ֻҪ����weight���� ��ʹ�� �����ʹ�� ��melt��ܲ���ȷ��weight����Ϊ�ռ���
			//���еķ�����  ��weight������attr ���� -weight 3 ���� -attr 3���Ե�����
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

		//@why bestStepRankingRegressionTrees ����ʲô��˼ ������� AdjustOutput��ʱ����ʹ�� _weightƽ�� ����LeastSquareRegression��ʹ����weight
		//default�Ĳ���ǡǡ�෴
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
			//gezi::zeroset(TrainScores); //��ӦBagging һ������ֵ������ ����Ϊ������ԭ�򡣡� ԭ������bagfraction �����1 ok
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
		//				//Fvec result = move(st->Scores); //���û��validating self ��ôֻ����һ�� ComputeScoresSmart(����clibrate)������move
		//				Fvec result = st->Scores;  
		//				return result;
		//			}
		//		}
		//	}
		//}

		//@FIXME for bagging with sampling fraction might better use orginal TrainSet/InoputInstances and their tracing score
		//���û��validate ԭʼ��Instance ��������һ�μ��㣿 ScoreTrackerû�м�¼
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
