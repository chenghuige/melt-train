/**
 *  ==============================================================================
 *
 *          \file   LibSVMTrainer.cpp
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-24 07:16:06.687935
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef LIB_S_V_M_TRAINER_CPP_
#define LIB_S_V_M_TRAINER_CPP_

#include "libsvm/svm.h"

#include "Trainers/LibSVMTrainer.h"
#include "Predictors/LibSVMPredictor.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

namespace gezi {

	namespace {
		struct svm_parameter thread_local param;		// set by parse_command_line
		struct svm_problem thread_local prob;		// set by read_problem
		struct svm_model thread_local *model;
		struct svm_node thread_local *x_space;
		int thread_local cross_validation;
		int thread_local nr_fold;

		void print_null(const char *s) {}

		void exit_with_help()
		{
			printf(
				"Usage: svm-train [options] training_set_file [model_file]\n"
				"options:\n"
				"-s svm_type : set type of SVM (default 0)\n"
				"	0 -- C-SVC		(multi-class classification)\n"
				"	1 -- nu-SVC		(multi-class classification)\n"
				"	2 -- one-class SVM\n"
				"	3 -- epsilon-SVR	(regression)\n"
				"	4 -- nu-SVR		(regression)\n"
				"-t kernel_type : set type of kernel function (default 2)\n"
				"	0 -- linear: u'*v\n"
				"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
				"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
				"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
				"	4 -- precomputed kernel (kernel values in training_set_file)\n"
				"-d degree : set degree in kernel function (default 3)\n"
				"-g gamma : set gamma in kernel function (default 1/num_features)\n"
				"-r coef0 : set coef0 in kernel function (default 0)\n"
				"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
				"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
				"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
				"-m cachesize : set cache memory size in MB (default 100)\n"
				"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
				"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
				"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
				"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
				"-v n: n-fold cross validation mode\n"
				"-q : quiet mode (no outputs)\n"
				);
			exit(1);
		}

		void parse_command_line(int argc, char **argv)
		{
			int i;
			void(*print_func)(const char*) = NULL;	// default printing to stdout

			// default values
			param.svm_type = C_SVC;
			param.kernel_type = RBF;
			param.degree = 3;
			param.gamma = 0;	// 1/num_features
			param.coef0 = 0;
			param.nu = 0.5;
			param.cache_size = 100;
			param.C = 1;
			param.eps = 1e-3;
			param.p = 0.1;
			param.shrinking = 1;
			param.probability = 0;
			param.nr_weight = 0;
			param.weight_label = NULL;
			param.weight = NULL;
			cross_validation = 0;

			// parse options
			for (i = 1; i < argc; i++)
			{
				if (argv[i][0] != '-') break;
				if (++i >= argc)
					exit_with_help();
				switch (argv[i - 1][1])
				{
				case 's':
					param.svm_type = atoi(argv[i]);
					break;
				case 't':
					param.kernel_type = atoi(argv[i]);
					break;
				case 'd':
					param.degree = atoi(argv[i]);
					break;
				case 'g':
					param.gamma = atof(argv[i]);
					break;
				case 'r':
					param.coef0 = atof(argv[i]);
					break;
				case 'n':
					param.nu = atof(argv[i]);
					break;
				case 'm':
					param.cache_size = atof(argv[i]);
					break;
				case 'c':
					param.C = atof(argv[i]);
					break;
				case 'e':
					param.eps = atof(argv[i]);
					break;
				case 'p':
					param.p = atof(argv[i]);
					break;
				case 'h':
					param.shrinking = atoi(argv[i]);
					break;
				case 'b':
					param.probability = atoi(argv[i]);
					break;
				case 'q':
					print_func = &print_null;
					i--;
					break;
				case 'v':
					cross_validation = 1;
					nr_fold = atoi(argv[i]);
					if (nr_fold < 2)
					{
						fprintf(stderr, "n-fold cross validation: n must >= 2\n");
						exit_with_help();
					}
					break;
				case 'w':
					++param.nr_weight;
					param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*param.nr_weight);
					param.weight = (double *)realloc(param.weight, sizeof(double)*param.nr_weight);
					param.weight_label[param.nr_weight - 1] = atoi(&argv[i - 1][2]);
					param.weight[param.nr_weight - 1] = atof(argv[i]);
					break;
				default:
					fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
					exit_with_help();
				}
			}

			svm_set_print_string_function(print_func);

			//// determine filenames

			//if (i >= argc)
			//	exit_with_help();

			//strcpy(input_file_name, argv[i]);

			//if (i < argc - 1)
			//	strcpy(model_file_name, argv[i + 1]);
			//else
			//{
			//	char *p = strrchr(argv[i], '/');
			//	if (p == NULL)
			//		p = argv[i];
			//	else
			//		++p;
			//	sprintf(model_file_name, "%s.model", p);
			//}

		}

		bool ParseCommandLine(int argc, char** argv)
		{
			parse_command_line(argc, argv);
			return true;
		}

		vector<svm_node> Instance2SvmNodeVec(InstancePtr instance)
		{
			vector<svm_node> vec;
			instance->features.ForEachNonZero([&](int index, Float value) {
				svm_node node;
				node.index = index + 1;
				node.value = value;
				vec.emplace_back(move(node));
			});
			svm_node node;
			node.index = -1;
			vec.emplace_back(move(node));
			return vec;
		}

	}


	void LibSVMTrainer::ShowHelp()
	{
		exit_with_help();
	}

	PredictorPtr LibSVMTrainer::CreatePredictor()
	{
		return make_shared<LibSVMPredictor>(model, &prob, x_space, &param, _normalizer, _calibrator, _featureNames);
	}

	svm_problem LibSVMTrainer::Instances2SvmProblem(Instances& instances)
	{
		Notifer timer("Instances2SvmProblem");

		svm_problem prob;
		prob.l = 0;

		int max_index, inst_max_index, i;
		long int elements, j;
		elements = 0;
		for (InstancePtr instance : instances)
		{
			//elements += instance->features.NumNonZeros();
			elements += instance->features.Count();
			elements++; //for end index of -1
			prob.l++;
		}

		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(struct svm_node *, prob.l);
		x_space = Malloc(struct svm_node, elements);

		max_index = 0;
		j = 0;
		for (i = 0; i < prob.l; i++)
		{
			inst_max_index = 0; // strtol gives 0 if wrong format
			prob.x[i] = &x_space[j];
			prob.y[i] = instances[i]->label;

			instances[i]->features.ForEachNonZero([&](int index, Float value) {
				x_space[j].index = index + 1;
				if (x_space[j].index > inst_max_index)
				{
					inst_max_index = x_space[j].index;
				}
				x_space[j].value = value;
				j++;
			});

			if (inst_max_index > max_index)
				max_index = inst_max_index;

			x_space[j++].index = -1;
		}

		if (param.gamma == 0 && max_index > 0)
			param.gamma = 1.0 / max_index;

		if (param.kernel_type == PRECOMPUTED)
			for (i = 0; i < prob.l; i++)
			{
				if (prob.x[i][0].index != 0)
				{
					fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
					exit(1);
				}
				if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
				{
					fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
					exit(1);
				}
			}

		return prob;
	}

	void LibSVMTrainer::Initialize(Instances& instances)
	{
		static String2ArgcArgv args("libsvm " + _classiferSettings);
		static bool inited = ParseCommandLine(args.argc(), args.argv());
	}

	void LibSVMTrainer::InnerTrain(Instances& instances)
	{
		prob = Instances2SvmProblem(instances);

		const char *error_msg = svm_check_parameter(&prob, &param);
		if (error_msg)
		{
			fprintf(stderr, "ERROR: %s\n", error_msg);
			exit(1);
		}

		{
			Notifer timer("LibSVM train");
			model = svm_train(&prob, &param);
			Pval2(prob.l, model->l);
		}
	}

	Float LibSVMTrainer::Margin(InstancePtr instance)
	{
		vector<svm_node> vec = Instance2SvmNodeVec(instance);
		svm_node* x = vec.empty() ? NULL : &vec[0];
		vector<double> probs(model->nr_class, 0);
		svm_predict_probability(model, x, &probs[0]);
		return probs[1];
	}

	void LibSVMTrainer::Finalize_(Instances& instances)
	{
		//svm_free_and_destroy_model(&model);
		//svm_destroy_param(&param);
		//free(prob.y);
		//free(prob.x); //model内部还依赖这个 不能free 放到predictor中析构 需要传递prob过去
		//free(x_space);
	}

}  //----end of namespace gezi

#endif  //----end of LIB_S_V_M_TRAINER_CPP_
