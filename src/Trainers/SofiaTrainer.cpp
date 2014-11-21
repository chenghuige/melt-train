/**
 *  ==============================================================================
 *
 *          \file   SofiaTrainer.cpp
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 07:19:23.881546
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef SOFIA_TRAINER_CPP_
#define SOFIA_TRAINER_CPP_

#include "sofia/simple-cmd-line-helper.h"
#include "Trainers/SofiaTrainer.h"

namespace gezi {

	//copy from sofia-ml.cc
	void CommandLine(int argc, char** argv, int start = 0) {
		//chg added 不考虑多线程 因为目前多个相同类型Trainer只是cross validation时候可能 而cross validation是顺序执行的
		static bool isFirst = true;
		if (!isFirst)
		{
			return;
		}
		else
		{
			isFirst = false;
		}
		AddFlag("--training_file", "File to be used for training.", string(""));
		AddFlag("--test_file", "File to be used for testing.", string(""));
		AddFlag("--results_file", "File to which to write predictions.", string(""));
		AddFlag("--model_in", "Read in a model from this file.", string(""));
		AddFlag("--model_out", "Write the model to this file.", string(""));
		AddFlag("--random_seed",
			"When set to non-zero value, use this seed instead of seed from system clock.\n"
			"    This can be useful for parameter tuning in cross-validation, as setting \n"
			"    a seed by hand forces examples to be sampled in the same order.  However\n"
			"    for actual training/test, this should never be used.\n"
			"    Default: 0",
			int(0));
		AddFlag("--lambda",
			"Value of lambda for svm regularization.\n"
			"    Default: 0.1",
			float(0.1));
		AddFlag("--iterations",
			"Number of stochastic gradient steps to take.\n"
			"    Default: 100000",
			int(100000));
		AddFlag("--learner_type",
			"Type of learner to use.\n"
			"    Options are: pegasos, passive-aggressive, margin-perceptron, "
			"romma, sgd-svm, least-mean-squares, logreg, and logreg-pegasos.\n"
			"    Default: pegasos",
			string("pegasos"));
		AddFlag("--eta_type",
			"Type of update for learning rate to use.\n"
			"    Options are: basic, pegasos, constant (0.02).\n"
			"    Default: pegasos",
			string("pegasos"));
		AddFlag("--loop_type",
			"Type of loop to use for training, controlling how examples are selected.\n"
			"    Options are: stochastic, balanced-stochastic, "
			"roc, rank, query-norm-rank, combined-ranking, "
			"combined-roc\n"
			"    Default: stochastic",
			string("stochastic"));
		AddFlag("--prediction_type",
			"Type of predictions to compute on test data.\n"
			"    Options are: linear, logistic\n"
			"    Use linear for SVM predictions.\n"
			"    Default: linear",
			string("linear"));
		AddFlag("--rank_step_probability",
			"Probability that we will take a rank step (as opposed to a  \n"
			"    classification step) in a combined-ranking or "
			"    combined-roc loop.\n"
			"    Default: 0.5",
			float(0.5));
		AddFlag("--passive_aggressive_c",
			"Maximum size of any step taken in a single passive-aggressive update.",
			float(10000000.0));
		AddFlag("--passive_aggressive_lambda",
			"Lambda for pegasos-style projection for passive-aggressive update.\n"
			"    When set to 0 (default) no projection is performed.",
			float(0));
		AddFlag("--perceptron_margin_size",
			"Width of margin for perceptron with margins.\n"
			"    Default of 1 is equivalent to un-regularized SVM-loss.",
			float(1.0));
		AddFlag("--training_objective",
			"Compute value of objective function on training data, after training.\n"
			"    Default is not to do this.",
			bool(false));
		AddFlag("--buffer_mb",
			"Size of buffer to use in reading/writing to files, in MB.\n"
			"    Default: 40",
			int(40));
		AddFlag("--dimensionality",
			"Index value of largest feature index in training data set. \n"
			"    Default: 2^17 = 131072",
			int(2 << 16));
		AddFlag("--hash_mask_bits",
			"When set to a non-zero value, causes the use of a hashed weight vector\n"
			"    with hashed cross product features.  The size of the hash table is set\n"
			"    to 2^--hash_mask_bits.  The same value of this flag must be used for\n"
			"    testing and training.\n"
			"    Default value of 0 shows that hash cross products are not used.",
			int(0));
		AddFlag("--no_bias_term",
			"When set, causes a bias term x_0 to be set to 0 for every \n"
			"    feature vector loaded from files, rather than the default of x_0 = 1.\n"
			"    This is equivalent to forcing a decision threshold of exactly 0 to be used.\n"
			"    Same setting of this flag should be used for training and testing. Note that\n"
			"    this flag as no effect for rank and roc optimzation.\n"
			"    Default: not set.",
			bool(false));
		ParseFlags(argc, argv, start);
	}

	void PrintElapsedTime(clock_t start, const string& message) {
		float num_secs = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
		std::cout << message << num_secs << std::endl;
	}

	void TrainModel(const SfDataSet& training_data, SfWeightVector* w) {
		clock_t train_start = clock();
		assert(w != NULL);

		// Default values.
		float lambda = CMD_LINE_FLOATS["--lambda"];
		float c = 0.0;

		sofia_ml::EtaType eta_type;
		if (CMD_LINE_STRINGS["--eta_type"] == "basic")
			eta_type = sofia_ml::BASIC_ETA;
		else if (CMD_LINE_STRINGS["--eta_type"] == "pegasos")
			eta_type = sofia_ml::PEGASOS_ETA;
		else if (CMD_LINE_STRINGS["--eta_type"] == "constant")
			eta_type = sofia_ml::CONSTANT;
		else {
			std::cerr << "--eta type " << CMD_LINE_STRINGS["--eta_type"] << " not supported.";
			exit(0);
		}

		sofia_ml::LearnerType learner_type;
		if (CMD_LINE_STRINGS["--learner_type"] == "pegasos")
			learner_type = sofia_ml::PEGASOS;
		else if (CMD_LINE_STRINGS["--learner_type"] == "margin-perceptron") {
			learner_type = sofia_ml::MARGIN_PERCEPTRON;
			c = CMD_LINE_FLOATS["--perceptron_margin_size"];
		}
		else if (CMD_LINE_STRINGS["--learner_type"] == "passive-aggressive") {
			learner_type = sofia_ml::PASSIVE_AGGRESSIVE;
			c = CMD_LINE_FLOATS["--passive_aggressive_c"];
			lambda = CMD_LINE_FLOATS["--passive_aggressive_lambda"];
		}
		else if (CMD_LINE_STRINGS["--learner_type"] == "logreg-pegasos")
			learner_type = sofia_ml::LOGREG_PEGASOS;
		else if (CMD_LINE_STRINGS["--learner_type"] == "logreg")
			learner_type = sofia_ml::LOGREG;
		else if (CMD_LINE_STRINGS["--learner_type"] == "least-mean-squares")
			learner_type = sofia_ml::LMS_REGRESSION;
		else if (CMD_LINE_STRINGS["--learner_type"] == "sgd-svm")
			learner_type = sofia_ml::SGD_SVM;
		else if (CMD_LINE_STRINGS["--learner_type"] == "romma")
			learner_type = sofia_ml::ROMMA;
		else {
			std::cerr << "--learner_type " << CMD_LINE_STRINGS["--learner_type"]
				<< " not supported.";
			exit(0);
		}

		if (CMD_LINE_STRINGS["--loop_type"] == "stochastic")
			sofia_ml::StochasticOuterLoop(training_data,
			learner_type,
			eta_type,
			lambda,
			c,
			CMD_LINE_INTS["--iterations"],
			w);
		else if (CMD_LINE_STRINGS["--loop_type"] == "balanced-stochastic")
			sofia_ml::BalancedStochasticOuterLoop(training_data,
			learner_type,
			eta_type,
			lambda,
			c,
			CMD_LINE_INTS["--iterations"],
			w);
		else if (CMD_LINE_STRINGS["--loop_type"] == "roc")
			sofia_ml::StochasticRocLoop(training_data,
			learner_type,
			eta_type,
			lambda,
			c,
			CMD_LINE_INTS["--iterations"],
			w);
		else if (CMD_LINE_STRINGS["--loop_type"] == "rank")
			sofia_ml::StochasticRankLoop(training_data,
			learner_type,
			eta_type,
			lambda,
			c,
			CMD_LINE_INTS["--iterations"],
			w);
		else if (CMD_LINE_STRINGS["--loop_type"] == "combined-ranking")
			sofia_ml::StochasticClassificationAndRankLoop(
			training_data,
			learner_type,
			eta_type,
			lambda,
			c,
			CMD_LINE_FLOATS["--rank_step_probability"],
			CMD_LINE_INTS["--iterations"],
			w);
		else if (CMD_LINE_STRINGS["--loop_type"] == "combined-roc")
			sofia_ml::StochasticClassificationAndRocLoop(
			training_data,
			learner_type,
			eta_type,
			lambda,
			c,
			CMD_LINE_FLOATS["--rank_step_probability"],
			CMD_LINE_INTS["--iterations"],
			w);
		else if (CMD_LINE_STRINGS["--loop_type"] == "query-norm-rank")
			sofia_ml::StochasticQueryNormRankLoop(training_data,
			learner_type,
			eta_type,
			lambda,
			c,
			CMD_LINE_INTS["--iterations"],
			w);
		else {
			std::cerr << "--loop_type " << CMD_LINE_STRINGS["--loop_type"] << " not supported.";
			exit(0);
		}

		PrintElapsedTime(train_start, "Time to complete training: ");
	}

	//-------------from SofiaTrainer

	void SofiaTrainer::ShowHelp()
	{
		CommandLine(0, NULL);
		print_map(CMD_LINE_BOOLS);
		print_map(CMD_LINE_INTS);
		print_map(CMD_LINE_FLOATS);
		print_map(CMD_LINE_DESCRIPTIONS);
	}

	SfDataSet SofiaTrainer::Instances2SfDataSet(Instances& instances)
	{
		Notifer timer("Instances2SfDataSet");
		bool useBias = !CMD_LINE_BOOLS["--no_bias_term"];
		SfDataSet sfDataSet(useBias);
		vector<SfSparseVector>& vec = sfDataSet.vectors_;
		for (InstancePtr instance : instances)
		{
			SfSparseVector sfVec(useBias);
			sfVec.y_ = instance->label <= 0 ? -1 : instance->label;
			instance->features.ForEach([&](int index, Float value) {
				sfVec.PushPair(index + 1, value);
			});
			vec.emplace_back(sfVec);
		}
		return sfDataSet;
	}

	void SofiaTrainer::Initialize(Instances& instances)
	{
		int argc;
		char** argv = NULL;
		Pval(_classiferSettings);
		stringToArgcArgv(_classiferSettings, &argc, &argv);
		CommandLine(argc, argv);
		freeArgcArgv(argc, argv);
		CMD_LINE_INTS["--random_seed"] = _randSeed;
		if (CMD_LINE_INTS["--random_seed"] == 0) {
			srand(time(NULL));
		}
		else {
			std::cerr << "Using random_seed: " << CMD_LINE_INTS["--random_seed"] << std::endl;
			srand(CMD_LINE_INTS["--random_seed"]);
		}

		// Set up empty model with specified dimensionality.
		CMD_LINE_INTS["--dimensionality"] = _numFeatures + 1; //包含一位bias
	
		if (CMD_LINE_INTS["--hash_mask_bits"] == 0) {
			w = new SfWeightVector(CMD_LINE_INTS["--dimensionality"]);
		}
		else {
			w = new SfHashWeightVector(CMD_LINE_INTS["--hash_mask_bits"]);
		}
	}

	void SofiaTrainer::InnerTrain(Instances& instances)
	{
		SfDataSet sfDataSet = Instances2SfDataSet(instances);
		{
			Notifer timer("Sofia training");
			TrainModel(sfDataSet, w);
		}
	}

	void SofiaTrainer::Finalize(Instances& instances)
	{
		_bias = w->ValueOf(0);
		_weights.resize(_numFeatures, 0);
		int len = w->GetDimensions();

		for (int i = 1; i < len; i++)
		{
			_weights[i - 1] = w->ValueOf(i);
		}
		FREE(w);

		if (_calibrator != nullptr)
		{
			_calibrator->Train(instances, [this](InstancePtr instance) {
				return Margin(instance->features); });
		}
	}

}  //----end of namespace gezi

#endif  //----end of SOFIA_TRAINER_CPP_
