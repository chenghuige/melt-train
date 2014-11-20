/**
*  ==============================================================================
*
*          \file   Trainers/SVM/LinearSVM.h
*
*        \author   chenghuige
*
*          \date   2014-04-06 21:41:18.444778
*
*  \Description:
SVM (Pegasos-Linear)	/cl:LinearSVM	Fast primal-space stochastic gradient descent solver (with an optional projection step).
S. Shalev-Shwartz, Y. Singer, and N. Srebro.  Pegasos:
Primal Estimated sub-GrAdient SOlver for SVM. ICML-2007.	Projection step is off by default (resulting in a 1.5-2x speedup).
It is advisable to pre-normalize or turn off normalization for sparse data. (or pre-normalize via /c CreateInstances if normalization does help)
*  ==============================================================================
*/

#ifndef TRAINERS__S_V_M__LINEAR_S_V_M_H_
#define TRAINERS__S_V_M__LINEAR_S_V_M_H_

#include "ProgressBar.h"
#include "MLCore/IterativeTrainer.h"
#include "Prediction/Instances/Instances.h"
#include "Numeric/Vector/Vector.h"
#include "Numeric/Vector/WeightVector.h"
#include "Predictors/LinearPredictor.h"
#include "Prediction/Normalization/NormalizerFactory.h"
#include "Prediction/Calibrate/CalibratorFactory.h"
#include "Prediction/Instances/instances_util.h"
namespace gezi {

	class LinearSVM : public IterativeTrainer
	{
	public:
		LinearSVM()
		{
			ParseArgs();
		}

		enum class LoopType
		{
			Stochastic,
			BalancedStochastic,
			Roc
		};

		enum class TrainerType
		{
			Pegasos,
			PassiveAggressive,
			MarginPerceptron,
			Romma,
			SgdSVM,
			LeastMeanSquares,
			Logreg,
			LogregPegasos
		};

		map<string, LoopType> _loopTypes = {
			{ "stochastic", LoopType::Stochastic },
			{ "balance", LoopType::BalancedStochastic },
			{ "balanced", LoopType::BalancedStochastic },
			{ "balancedstochastic", LoopType::BalancedStochastic },
			{ "roc", LoopType::Roc },
		};

		map<string, TrainerType> _trainerTypes = {
			{ "pegasos", TrainerType::Pegasos },
			{ "passiveaggressive", TrainerType::PassiveAggressive },
			{ "marginperceptron", TrainerType::MarginPerceptron },
			{ "romma", TrainerType::Romma },
			{ "sgdsvm", TrainerType::SgdSVM },
			{ "leastmeansquares", TrainerType::LeastMeanSquares },
			{ "logreg", TrainerType::Logreg },
			{ "logregpegasos", TrainerType::LogregPegasos },
		};

		struct Arguments
		{
			int numIterations = 50000; //iter|Number of iterations
			Float lambda = 0.001; //lr| learning rate
			Float sampleRate = 0.001; //sr|Sampling rate
			int sampleSize = 1; //ss|Sampling size
			bool performProjection = false; //project|Perform projection to unit-ball
			bool noBias = false;
			string initialWeightsString = ""; //initweights|Initial weights and bias, comma-separated
			bool randomInitialWeights = false; //randweights|Randomize initial weights
			int featureNumThre = 1000; //fnt|if NumFeatures > featureNumThre use dense format 
			//暂时不支持streaming 模式
			bool doStreamingTraining = false; //stream|Streaming instances training

			bool normalizeFeatures = true; //norm|Normalize features
			string normalizerName = "MinMax"; //normalizer|Which normalizer?

			unsigned randSeed = 0;//rs|controls wether the expermient can reproduce, 0 means not reproduce

			bool calibrateOutput = true; //calibrate| use calibrator to gen probability?
			string calibratorName = "sigmoid"; //calibrator| sigmoid/platt naive pav
			//uint64 maxCalibrationExamples = 1000000; //numCali|Number of instances to train the calibrator

			string loopType = "stochastic"; //lt| now support [stochastic, balancedStochastic, roc] like sofia-ml will support [stochastic, balancedStochastic, roc, rank, queryNormRank, combinedRanking, combinedRoc]
			string trainerType = "pegasos"; //trt| now support [pegasos] like sofia-ml will support [pegasos, passiveAggressive, marginPerceptron, romma, sgdSvm, leastMeanSquares, logreg, and logregPegasos]
		};

		virtual void ShowHelp() override
		{
			fmt::print_line("DECLARE_bool(calibrate);");
			fmt::print_line("DECLARE_string(calibrator);");
			fmt::print_line("DECLARE_uint64(rs);");
			fmt::print_line("DECLARE_bool(norm); //will speed up a if pre normalize and then --norm=0 for cross validation");
			fmt::print_line("DECLARE_string(normalizer);");
			fmt::print_line("DEFINE_int32(iter, 50000, \"numIterExamples: Number of iterations\");");
			fmt::print_line("DEFINE_double(lr, 0.001, \"lambda: learning rate\");");
			fmt::print_line("DEFINE_string(lt, \"stochastic\", \"loopType: try roc or balanced\");");
			fmt::print_line("DEFINE_string(trt, \"peagsos\", \"trainerType: now only support peagsos\");");

			fmt::print_line("int numIterations = 50000; //iter|Number of iterations");
			fmt::print_line("Float lambda = 0.001; //lr|");
			fmt::print_line("Float sampleRate = 0.001; //sr|Sampling rate");
			fmt::print_line("int sampleSize = 1; //ss|Sampling size");
			fmt::print_line("bool performProjection = false; //project|Perform projection to unit-ball");
			fmt::print_line("bool noBias = false;");
			fmt::print_line("string initialWeightsString = \"\"; //initweights|Initial weights and bias, comma-separated");
			fmt::print_line("bool randomInitialWeights = false; //randweights|Randomize initial weights");
			fmt::print_line("int featureNumThre = 1000; //fnt|if NumFeatures > featureNumThre use dense format");
			fmt::print_line("bool doStreamingTraining = false; //stream|Streaming instances training");
			fmt::print_line("bool normalizeFeatures = true; //norm|Normalize features");
			fmt::print_line("string normalizerName = \"MinMax\"; //normalizer|Which normalizer?");
			fmt::print_line("unsigned randSeed = 0;//rs|controls wether the expermient can reproduce, 0 means not reproduce");
			fmt::print_line("bool calibrateOutput = true; //calibrate| use calibrator to gen probability?");
			fmt::print_line("string calibratorName = \"sigmoid\"; //calibrator| sigmoid/platt naive pav");
			fmt::print_line("sstring loopType = \"stochastic\"; //lt| now support [stochastic, balancedStochastic, roc] like sofia-ml will support [stochastic, balancedStochastic, roc, rank, queryNormRank, combinedRanking, combinedRoc]");
			fmt::print_line("string trainerType = \"pegasos\"; //trt| now support [pegasos] like sofia-ml will support [pegasos, passiveAggressive, marginPerceptron, romma, sgdSvm, leastMeanSquares, logreg, and logregPegasos]");
		}

		virtual string GetParam() override
		{
			stringstream ss;
			ss << "numIterations:" << _args.numIterations << " "
				<< "learningRate:" << _args.lambda << " "
				<< "trainerTyper:" << _args.trainerType << " "
				<< "loopType:" << _args.loopType << " "
				<< "sampleSize:" << _args.sampleSize << " "
				<< "performProjection:" << _args.performProjection;
			return ss.str();
		}

		virtual void ParseArgs() override;
		virtual void Init() override
		{
			PVAL(_args.randSeed);
			_rand = make_shared<Random>(random_engine(_args.randSeed));
			if (_args.normalizeFeatures) //@TODO to trainer
			{
				_normalizer = NormalizerFactory::CreateNormalizer(_args.normalizerName);
			}
			PVAL((_normalizer == nullptr));

			if (_args.calibrateOutput) //@TODO to trainer
			{
				_calibrator = CalibratorFactory::CreateCalibrator(_args.calibratorName);
			}
			PVAL((_calibrator == nullptr));
		}

		virtual void Initialize(Instances& instances) override
		{
			_sampleSize = _args.sampleSize == 0 ? instances.Count() * _args.sampleRate : _args.sampleSize;

			_numFeatures = instances.FeatureNum();
			_randRange = make_shared<RandomRange>(instances.Count(), random_engine(_args.randSeed));

			if (_args.initialWeightsString.size() > 0)
			{
				LOG(INFO) << "Initializing weights and bias to " << _args.initialWeightsString;
				svec weightStr = split(_args.initialWeightsString, ',');
				if ((int)weightStr.size() == _numFeatures + 1)
				{
					Fvec weightArr(_numFeatures);
					for (int i = 0; i < _numFeatures; i++)
						weightArr[i] = DOUBLE(weightStr[i]);
					_weights.Init(weightArr);
					_bias = DOUBLE(weightStr[_numFeatures]);
				}
				else
				{
					LOG(WARNING) << "Could not inialize weights and bias from input use default";
				}
			}

			// weight initialization -- done unless we have initialized before
			if (_weights.Length() == 0)
			{
				// We want a dense vector, to prevent memory creation during training
				// unless we have a lot of features
				_weights.SetLength(_numFeatures);
				_bias = 0;

				// weights may be set to random numbers distributed uniformly on -1,1
				if (_args.randomInitialWeights)
				{
					for (int featureIdx = 0; featureIdx < _numFeatures; featureIdx++)
					{
						//_weights[featureIdx] = 2 * _rand->NextFloat() - 1; //@FIXME
						_weights.values[featureIdx] = 2 * _rand->NextFloat() - 1;
					}
					if (!_args.noBias)
						_bias = 2 * _rand->NextFloat() - 1;
				}
			}

			// counters
			_iteration = 0;
			_numProcessedExamples = 0;
			_numIterExamples = 0;

			_featureNames = instances.schema.featureNames;

			VLOG(3) << "Initialized LinearSVM on " << _numFeatures << " features";
		}

		virtual void BeginTrainingIteration() override
		{
			_numProcessedExamples = 0;
			_numIterExamples = 0;
			_gradientUpdate = 0;

			_weightsUpdate.clear();
			_weightUpdates.clear();
			_biasUpdates.clear();
		}

		//@TODO move


		/// Override the default training loop:   we need to pick random instances manually...
		virtual void InnerTrain(Instances& instances_) override
		{
			//@TODO 兼容streaming模式
			if (_normalizer != nullptr && _normalizeCopy && !instances_.IsNormalized())
			{
				normalizedInstances() = _normalizer->NormalizeCopy(instances_);
				_instances = &normalizedInstances();
			}
			else
			{
				_instances = &instances_;
			}

			Instances& instances = *_instances;
			//ProgressBar pb(format("LinearSVM training with trainerType {}, loopType {}", _args.trainerType, _args.loopType), _args.numIterations);
			ProgressBar pb("LinearSVM training", _args.numIterations);
			//AutoTimer timer("LinearSVM training", 0);
			LoopType loopType = _loopTypes[arg(_args.loopType)];
			TrainerType trainerType = _trainerTypes[arg(_args.trainerType)];
			Instances posInstances, negInstances;
			if (loopType == LoopType::BalancedStochastic || loopType == LoopType::Roc)
			{
				InstancesUtil::SplitInstancesByLabel(instances, posInstances, negInstances);
			}
			for (int iter = 0; iter < _args.numIterations; iter++)
			{
				++pb;
				++_iteration;
				BeginTrainingIteration();
				if (loopType == LoopType::Stochastic || loopType == LoopType::BalancedStochastic)
				{
					for (int i = 0; i < _sampleSize; i++)
					{
						if (loopType == LoopType::Stochastic)
						{
							_currentIdx = _rand->Next(instances.Count());
							_currentInstance = instances[_currentIdx];
							ProcessDataInstance(_currentInstance);
						}
						else if (loopType == LoopType::BalancedStochastic)
						{
							if ((iter * _sampleSize + i) % 2 == 0)
							{
								_currentIdx = _rand->Next(posInstances.Count());
								_currentInstance = posInstances[_currentIdx];
							}
							else
							{
								_currentIdx = _rand->Next(negInstances.Count());
								_currentInstance = negInstances[_currentIdx];
							}
							ProcessDataInstance(_currentInstance);
						}
						//_currentIdx = _rand->Next(instances.Count());
						//_currentIdx = _randRange->Next();
						//_currentIdx = static_cast<int>(rand()) % instances.Count();
						//ProcessDataInstance(instances[_currentIdx]);
					}
					FinishDataIteration();
				}
				else if (loopType == LoopType::Roc)
				{
					InstancePtr posInstance, negInstance;
					posInstance = posInstances[_rand->Next(posInstances.Count())];
					negInstance = negInstances[_rand->Next(negInstances.Count())];
					//_currentInstance = make_shared<Instance>(*posInstance); 
					//_currentInstance->features -= negInstance->features;
					//ProcessDataInstance(_currentInstance);
					ProcessDataInstance(posInstance, negInstance);
				}
			}
		}

		virtual void Finalize(Instances& instances_) override
		{
			if (_calibrator != nullptr)
			{
				Instances& instances = *_instances;
				_calibrator->Train(instances, [this](InstancePtr instance) {
					//if (_normalizer != nullptr && !instance->normalized)
					//{//@TODO 等于重复做了一次normalize
					//	instance = _normalizer->NormalizeCopy(instance);
					//}
					return Margin(instance->features); });
			}
		}

		//ROC svm输入是正负两个label的向量，并且暂时仿照sofia不支持一次选多个 @TODO流程可以合并 设置RocInsatnce持有两个InstancePtr即可 需要函数模板
		//整合dotOnDifference和AddScale + AddScale统一接口
		//同时这里不考虑instance的weight
		void ProcessDataInstance(InstancePtr posInstance, InstancePtr negInstance)
		{
			++_numIterExamples;
			if (_normalizer != nullptr && _normalizeCopy)
			{//如果不需要normalizeCopy前面Inialize的时候统一都normalize了
				if (!posInstance->normalized)
				{
					posInstance = _normalizer->NormalizeCopy(posInstance);
				}
				if (!negInstance->normalized)
				{
					negInstance = _normalizer->NormalizeCopy(negInstance);
				}
			}

			Float output = _bias + _weights.dotOnDifference(posInstance->features, negInstance->features);
			Float trueOutput = (posInstance->label > negInstance->label) ? 1 : -1;
			_loss = 1 - output * trueOutput;

			//------------------------scale
			Float learningRate = 1 / (_args.lambda * _iteration);
			Float scale = 1 - learningRate * _args.lambda;

			if (scale <= 0.0000001)
			{ //来自sofia-ml
				//LOG(WARNING) << scale;
				scale = 0.0000001;
			}

			_weights.ScaleBy(scale);
			_bias *= scale;

			//---------------------update gradient
			if (_loss > 0)
			{
				Float update = trueOutput * learningRate / _numIterExamples;
				_weights.AddScale(posInstance->features, update);
				_weights.AddScale(negInstance->features, -update);
				if (!_args.noBias)
				{
					_bias += update;
				}
			}

			//---------------------project
			//@TODO check performProjection
			// w_{t+1} = min{1, 1/sqrt(lambda)/|w_{t+1/2}|} * w_{t+1/2}
			if (_args.performProjection)
			{
				Float normalizer = 1 / sqrt(_args.lambda * _weights.squaredNorm);
				Pval2_4(normalizer, _weights.squaredNorm);
				if (normalizer < 1)
				{
					_weights.ScaleBy(normalizer);
					//_bias = _bias * normalizer; //@TODO tlc注释了这个？ sofia用统一向量 貌似都有*吧 需要看论文确认
				}
			}
		}

		/// Observe an example and update weights if necessary
		void ProcessDataInstance(InstancePtr instance)
		{
			++_numIterExamples;

			//if (_normalizer != nullptr && !instance->normalized && _normalizeCopy)
			//{//如果不需要normalizeCopy前面Inialize的时候统一都normalize了
			//	instance = _normalizer->NormalizeCopy(instance);
			//}

			_currentInstance = instance;

			// compute the update and update if needed     
			Float output = Margin(instance->features);
			Float trueOutput = (instance->IsPositive() ? 1 : -1);
			_loss = 1 - output * trueOutput;

			// record the update if there is a loss
			if (_loss > 0)
			{
				Float currentBiasUpdate;
				Vector currentWeightUpdate;
				GetUpdate(output, trueOutput, instance,
					ref(currentWeightUpdate), ref(currentBiasUpdate));

				if (_sampleSize == 1)
				{
					_gradientUpdate = currentBiasUpdate;
				}
				else
				{
					if (_args.sampleSize == 0)
					{ // rate sampling                                   
						if (_weightsUpdate.empty())
						{
							_weightsUpdate = move(currentWeightUpdate);
						}
						else
						{
							_weightsUpdate.Add(currentWeightUpdate);
						}
						_gradientUpdate += currentBiasUpdate;
					}
					else
					{ // pick a slot
						if (_weightUpdates.size() < _args.sampleSize)
						{
							_weightUpdates.emplace_back(currentWeightUpdate);
							_biasUpdates.push_back(currentBiasUpdate);
						}
						else
						{ // need to replace random one
							//int idxToReplace = _randRange->Next(_args.sampleSize);
							int idxToReplace = _randRange->Next();
							_weightUpdates[idxToReplace] = move(currentWeightUpdate);
							_biasUpdates[idxToReplace] = currentBiasUpdate;
						}
					}
				}
			}
		}

		void FinishDataIteration()
		{
			if (_numIterExamples > 0)
			{
				ScaleWeights();
			}

			_numProcessedExamples = 0;
			_numIterExamples = 0;
		}

		Arguments& Args()
		{
			return _args;
		}

		virtual PredictorPtr CreatePredictor() override
		{
			return make_shared<LinearPredictor>(_weights.ToVector(), _bias,
				_normalizer, _calibrator,
				_featureNames,
				"LinearSVM");
		}

	protected:
	private:
		/// <summary>
		/// Return the raw margin from the decision hyperplane
		/// </summary>		
		Float Margin(const Vector& features)
		{
			return _bias + _weights.dot(features);
		}

		// <summary>
		/// Given an impression, and the output of the classifier, compute an update
		/// </summary>        
		void GetUpdate(Float output, Float trueOutput, InstancePtr instance,
			Vector& gradient, Float& _gradientUpdate)
		{
			// scale regret by weight
			_gradientUpdate = trueOutput * instance->weight;
			if (_sampleSize > 1)
			{
				gradient = instance->features;
				gradient.ScaleBy(_gradientUpdate);
			}
		}

		/// <summary>
		/// Scale the weights at the end of the iteration
		/// </summary>
		void ScaleWeights()
		{
			if (_args.sampleSize == 0)
			{ // rate sampling
				ScaleWeightsSampled();
			}
			else
			{ // size sampling  走这里
				ScaleWeightsFixed();
			}
		}

		/// <summary>
		/// Scale the weights at the end of the iteration when we're sampling training instances
		/// </summary>
		void ScaleWeightsSampled()
		{
			// w_{t+1/2} = (1-eta*lambda) w_t + eta/k * totalUpdate
			Float learningRate = 1 / (_args.lambda * _iteration);
			Float scale = 1 - learningRate * _args.lambda;

			if (scale <= 0.0000001)
			{ //来自sofia-ml
				//LOG(WARNING) << scale;
				scale = 0.0000001;
			}

			_weights.ScaleBy(scale);
			_bias *= scale;

			if (_loss > 0)
			{
				if (_sampleSize == 1)
				{
					Float update = _gradientUpdate * learningRate / _numIterExamples;
					_weights.AddScale(_currentInstance->features, update);
					if (!_args.noBias)
					{
						_bias += update;
					}
				}
				else
				{
					Float update = learningRate / _numIterExamples;
					_weights.AddScale(_weightsUpdate, update);
					if (!_args.noBias)
					{
						_bias += _gradientUpdate * update;
					}
				}
			}

			//@TODO check performProjection
			// w_{t+1} = min{1, 1/sqrt(lambda)/|w_{t+1/2}|} * w_{t+1/2}
			if (_args.performProjection)
			{
				Float normalizer = 1 / sqrt(_args.lambda * _weights.squaredNorm);
				Pval2_4(normalizer, _weights.squaredNorm);
				if (normalizer < 1)
				{
					_weights.ScaleBy(normalizer);
					//_bias = _bias * normalizer; //@TODO tlc注释了这个？ sofia用统一向量 貌似都有*吧 需要看论文确认
				}
			}
		}

		/// <summary>
		/// Scale the weights at the end of the iteration when we picked a random number of training instances
		/// </summary>
		void ScaleWeightsFixed()
		{
			if (_sampleSize > 1)
			{
				// add up the updates
				for (Vector& nextUpdate : _weightUpdates)
				{
					if (_weightsUpdate.Empty())
					{
						_weightsUpdate = move(nextUpdate);
					}
					else
					{
						_weightsUpdate.Add(nextUpdate);
					}
				}
				// add up bias update
				for (Float bUpdate : _biasUpdates)
				{
					_gradientUpdate += bUpdate;
				}
			}

			ScaleWeightsSampled();
		}

	private:
		Arguments _args;

		FeatureNamesVector _featureNames;

		/// <summary> Total number of features </summary>
		int _numFeatures;
		/// <summary> Feature weights: weights for the last-seen training example </summary>
		//Vector _weights;
		WeightVector _weights;

		/// <summary> Prediction bias </summary>
		/// TODO: Note, I changed this also to mean the averaged bias. Should probably have two functions to
		///  make explicit whether you want the averaged or last bias. Same for weights.
		Float _bias = 1.; //初始1 原来0？ @TODO

		int _sampleSize = 1;

		InstancePtr _currentInstance = nullptr;
		Float _loss = 0;

		// number of processed examples and actual weight updates
		uint64 _numProcessedExamples = 0;
		uint64 _numIterExamples = 0;

		int _iteration = 0;

		Vector _weightsUpdate;
		Float _gradientUpdate = 0;
		vector<Vector> _weightUpdates;
		Fvec _biasUpdates;

		int64 _currentIdx = 0;
		int64 _lastIdx = -1;

		static Instances& normalizedInstances()
		{
			static thread_local Instances _normalizedInstances;
			return _normalizedInstances;
		}
		Instances* _instances = NULL;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__S_V_M__LINEAR_S_V_M_H_

