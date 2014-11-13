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
namespace gezi {

	class LinearSVM : public IterativeTrainer
	{
	public:
		LinearSVM()
		{

		}

		struct Arguments
		{
			int numIterations = 50000; //iter|Number of iterations
			Float lambda = 0.001; //lr|
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
		};

		void ParseArgs();

		virtual string GetParam() override
		{
			stringstream ss;
			ss << "numIterations:" << _args.numIterations << " "
				<< "learningRate:" << _args.lambda;
			return ss.str();
		}

		void Init()
		{
			ParseArgs();
			PVAL(_args.randSeed);
			_rand = make_shared<Random>(random_engine(_args.randSeed));
			if (_args.normalizeFeatures) //@TODO to trainer
			{
				_normalizer = NormalizerFactory::CreateNormalizer(_args.normalizerName);
			}
			PVAL((_normalizer == nullptr));

		}

		virtual void Initialize(Instances& instances) override
		{

			Init();

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

			//--- 将所有数据归一化 和TLC策略不同 TLC将normalize混在训练过程中(主要可能是兼容streaming模式)
			//特别是hadoop scope训练  @TODO  也许这里也会变化
			//如果不是类似交叉验证 比如就是训练测试 默认是 no normalize copy
			if (_normalizer != nullptr && !instances.IsNormalized())
			{
				if (!_normalizeCopy)
					_normalizer->RunNormalize(instances);
				else
					_normalizer->Prepare(instances);
			}

			VLOG(3) << "Initialized LinearSVM on " << _numFeatures << " features";
		}

		virtual void BeginTrainingIteration() override
		{
			_numProcessedExamples = 0;
			_numIterExamples = 0;
			_biasUpdate = 0;

			_weightsUpdate.clear();
			_weightUpdates.clear();
			_biasUpdates.clear();
		}

		/// Override the default training loop:   we need to pick random instances manually...
		virtual void InnerTrain(Instances& instances) override
		{
			_featureNames = instances.schema.featureNames;
			ProgressBar pb("LinearSVM training", _args.numIterations);
			//AutoTimer timer("LinearSVM training", 0);
			for (int iter = 0; iter < _args.numIterations; iter++)
			{
				++pb;
				BeginTrainingIteration();

				for (int i = 0; i < _sampleSize; i++)
				{
					_currentIdx = _rand->Next(instances.Count());
					//_currentIdx = _randRange->Next();
					//_currentIdx = static_cast<int>(rand()) % instances.Count();
					ProcessDataInstance(instances[_currentIdx]);
				}

				FinishDataIteration();
			}

			TrainingComplete();
		}

		virtual void Finalize(Instances& instances) override
		{
			if (_args.calibrateOutput) //@TODO to trainer
			{
				_calibrator = CalibratorFactory::CreateCalibrator(_args.calibratorName);
			}
			PVAL((_calibrator == nullptr));
			if (_calibrator)
			{
				_calibrator->Train(instances, [this](InstancePtr instance) {
					if (_normalizer != nullptr && !instance->normalized)
					{
						instance = _normalizer->NormalizeCopy(instance);
					}
					return Margin(instance->features); });
			}
		}

		/// Observe an example and update weights if necessary
		bool ProcessDataInstance(InstancePtr instance)
		{
			++_numIterExamples;

			if (_normalizer != nullptr && !instance->normalized && _normalizeCopy)
			{
				instance = _normalizer->NormalizeCopy(instance);
			}

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
					_biasUpdate = currentBiasUpdate;
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
						_biasUpdate += currentBiasUpdate;
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

			return true;
		}

		void FinishDataIteration()
		{
			++_iteration;

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
			Vector& gradient, Float& biasUpdate)
		{
			// scale regret by weight
			trueOutput *= instance->weight;
			if (_sampleSize > 1)
			{
				gradient = instance->features;
				gradient.ScaleBy(trueOutput);
			}
			biasUpdate = _args.noBias ? 0 : trueOutput;
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
					Float update = _biasUpdate * learningRate / _numIterExamples;
					_weights.AddScale(_currentInstance->features, update);
					_bias += update;
				}
				else
				{
					Float update = learningRate / _numIterExamples;
					_weights.AddScale(_weightsUpdate, update);
					_bias += _biasUpdate * update;
				}
			}

			//@TODO check performProjection
			// w_{t+1} = min{1, 1/sqrt(lambda)/|w_{t+1/2}|} * w_{t+1/2}
			if (_args.performProjection)
			{
				Float normalizer = 1 / sqrt(_args.lambda * _weights.squaredNorm); //@FIXME WeightVector::Norm() like sofia
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
					_biasUpdate += bUpdate;
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
		Float _bias;

		int _sampleSize = 1;

		InstancePtr _currentInstance = nullptr;
		Float _loss = 0;

		// number of processed examples and actual weight updates
		uint64 _numProcessedExamples = 0;
		uint64 _numIterExamples = 0;

		int _iteration = 0;

		Vector _weightsUpdate;
		Float _biasUpdate = 0;
		vector<Vector> _weightUpdates;
		Fvec _biasUpdates;

		int64 _currentIdx = 0;
		int64 _lastIdx = -1;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__S_V_M__LINEAR_S_V_M_H_

