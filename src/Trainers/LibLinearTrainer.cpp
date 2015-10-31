/**
 *  ==============================================================================
 *
 *          \file   LibLinearTrainer.cpp
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-23 14:54:17.543888
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef LIB_LINEAR_TRAINER_CPP_
#define LIB_LINEAR_TRAINER_CPP_

#include "Trainers/LibLinearTrainer.h"
#include "liblinear/linear.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#include "common_util.h"
DECLARE_int32(nt);

namespace gezi {

	namespace
	{
		struct feature_node *x_space;
		struct parameter param;
		struct problem prob;
		struct model* model_;
		int flag_cross_validation;
		int flag_find_C;
		int flag_omp;
		int flag_C_specified;
		int flag_solver_specified;
		int nr_fold;
		double bias;

		void print_null(const char *s) {}

		void exit_with_help()
		{
			printf(
				"Usage: train [options] training_set_file [model_file]\n"
				"options:\n"
				"-s type : set type of solver (default 1)\n"
				"  for multi-class classification\n"
				"	 0 -- L2-regularized logistic regression (primal)\n"
				"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
				"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
				"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
				"	 4 -- support vector classification by Crammer and Singer\n"
				"	 5 -- L1-regularized L2-loss support vector classification\n"
				"	 6 -- L1-regularized logistic regression\n"
				"	 7 -- L2-regularized logistic regression (dual)\n"
				"  for regression\n"
				"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
				"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
				"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
				"-c cost : set the parameter C (default 1)\n"
				"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
				"-e epsilon : set tolerance of termination criterion\n"
				"	-s 0 and 2\n"
				"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
				"		where f is the primal function and pos/neg are # of\n"
				"		positive/negative data (default 0.01)\n"
				"	-s 11\n"
				"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
				"	-s 1, 3, 4, and 7\n"
				"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
				"	-s 5 and 6\n"
				"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
				"		where f is the primal function (default 0.01)\n"
				"	-s 12 and 13\n"
				"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
				"		where f is the dual function (default 0.1)\n"
				"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
				"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
				"-v n: n-fold cross validation mode\n"
				"-C : find parameter C (only for -s 0 and 2)\n"
				"-n nr_thread : parallel version with [nr_thread] threads (default 1; only for -s 0, 2, 11)\n"
				"-q : quiet mode (no outputs)\n"
				);
			exit(1);
		}

		//void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
		void parse_command_line(int argc, char **argv)
		{
			int i;
			void(*print_func)(const char*) = NULL;	// default printing to stdout

			// default values
			param.solver_type = L2R_L2LOSS_SVC_DUAL;
			param.C = 1;
			param.eps = INF; // see setting below
			param.p = 0.1;
			param.nr_thread = FLAGS_nt;
			param.nr_weight = 0;
			param.weight_label = NULL;
			param.weight = NULL;
			param.init_sol = NULL;
			flag_cross_validation = 0;
			flag_C_specified = 0;
			flag_solver_specified = 0;
			flag_find_C = 0;
			flag_omp = (FLAGS_nt > 1);
			bias = -1;

			// parse options
			for (i = 1; i < argc; i++)
			{
				if (argv[i][0] != '-') break;
				if (++i >= argc)
					exit_with_help();
				switch (argv[i - 1][1])
				{
				case 's':
					param.solver_type = atoi(argv[i]);
					flag_solver_specified = 1;
					break;

				case 'c':
					param.C = atof(argv[i]);
					flag_C_specified = 1;
					break;

				case 'p':
					param.p = atof(argv[i]);
					break;

				case 'e':
					param.eps = atof(argv[i]);
					break;

				case 'B':
					bias = atof(argv[i]);
					break;

				case 'n':
					flag_omp = 1;
					param.nr_thread = atoi(argv[i]);
					break;

				case 'w':
					++param.nr_weight;
					param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*param.nr_weight);
					param.weight = (double *)realloc(param.weight, sizeof(double)*param.nr_weight);
					param.weight_label[param.nr_weight - 1] = atoi(&argv[i - 1][2]);
					param.weight[param.nr_weight - 1] = atof(argv[i]);
					break;

				case 'v':
					flag_cross_validation = 1;
					nr_fold = atoi(argv[i]);
					if (nr_fold < 2)
					{
						fprintf(stderr, "n-fold cross validation: n must >= 2\n");
						exit_with_help();
					}
					break;

				case 'q':
					print_func = &print_null;
					i--;
					break;

				case 'C':
					flag_find_C = 1;
					i--;
					break;

				default:
					fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
					exit_with_help();
					break;
				}
			}

			set_print_string_function(print_func);

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


			// default solver for parameter selection is L2R_L2LOSS_SVC
			if (flag_find_C)
			{
				if (!flag_cross_validation)
					nr_fold = 5;
				if (!flag_solver_specified)
				{
					fprintf(stderr, "Solver not specified. Using -s 2\n");
					param.solver_type = L2R_L2LOSS_SVC;
				}
				else if (param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC)
				{
					fprintf(stderr, "Warm-start parameter search only available for -s 0 and -s 2\n");
					exit_with_help();
				}
			}

			//default solver for parallel execution is L2R_L2LOSS_SVC
			if (flag_omp)
			{
				if (!flag_solver_specified)
				{
					fprintf(stderr, "Solver not specified. Using -s 2\n");
					param.solver_type = L2R_L2LOSS_SVC;
				}
				else if (param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC && param.solver_type != L2R_L2LOSS_SVR)
				{
					fprintf(stderr, "Parallel LIBLINEAR is only available for -s 0, 2, 11 now\n");
					exit_with_help();
				}
#ifdef CV_OMP
				omp_set_nested(1);
				omp_set_num_threads(nr_fold);
				if (nr_fold*param.nr_thread > omp_get_max_threads())
					fprintf(stderr, "The number of threads exceeds maxminum limit\n");
				else
					printf("Total threads used: %d\n", nr_fold*param.nr_thread);
#else
				printf("Total threads used: %d\n", param.nr_thread);
#endif
			}
#ifdef CV_OMP
			else
			{
				omp_set_num_threads(nr_fold);
				printf("Total threads used: %d\n", nr_fold);
			}
#endif

			if (param.eps == INF)
			{
				switch (param.solver_type)
				{
				case L2R_LR:
				case L2R_L2LOSS_SVC:
					param.eps = 0.01;
					break;
				case L2R_L2LOSS_SVR:
					param.eps = 0.001;
					break;
				case L2R_L2LOSS_SVC_DUAL:
				case L2R_L1LOSS_SVC_DUAL:
				case MCSVM_CS:
				case L2R_LR_DUAL:
					param.eps = 0.1;
					break;
				case L1R_L2LOSS_SVC:
				case L1R_LR:
					param.eps = 0.01;
					break;
				case L2R_L1LOSS_SVR_DUAL:
				case L2R_L2LOSS_SVR_DUAL:
					param.eps = 0.1;
					break;
				}
			}
		}

		void* ParseCommandLine(int argc, char** argv)
		{
			parse_command_line(argc, argv);
			return NULL;
		}

	}

	void LibLinearTrainer::ShowHelp()
	{
		exit_with_help();
	}

	problem LibLinearTrainer::Instances2Problem(Instances& instances)
	{
		Notifer timer("Instances2Problem");
	
		problem prob;
		prob.l = 0;
	
		int max_index, inst_max_index, i;
		long int elements, j;
		elements = 0;
		for (InstancePtr instance : instances)
		{
			//elements += instance->features.NumNonZeros(); //没太大必要吧，直接都用Count就好了吧 更安全
			elements += instance->features.Count();
			elements++; //for bias
			prob.l++;
		}
		prob.bias = bias;

		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(struct feature_node *, prob.l);
		x_space = Malloc(struct feature_node, elements + prob.l); // = prob.l for end index of -1
		//Pval3(elements, prob.l, (elements + prob.l));
		max_index = 0;
		j = 0;
		for (i = 0; i < prob.l; i++)
		{
			inst_max_index = 0; // strtol gives 0 if wrong format
			prob.x[i] = &x_space[j];
			prob.y[i] = instances[i]->label;

			instances[i]->features.ForEachNonZero([&](int index, Float value) { 
				//CHECK_LT(j, elements + prob.l) << i;
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

			if (prob.bias >= 0)
				x_space[j++].value = prob.bias;

			x_space[j++].index = -1;
		}

		if (prob.bias >= 0)
		{
			prob.n = max_index + 1;
			for (i = 1; i < prob.l; i++)
				(prob.x[i] - 2)->index = prob.n;
			x_space[j - 2].index = prob.n;
		}
		else
			prob.n = max_index;

		return prob;
	}

	void LibLinearTrainer::Initialize(Instances& instances)
	{
		static String2ArgcArgv args("liblinear " + _classiferSettings);
		static bool inited = ParseCommandLine(args.argc(), args.argv());
	}
	
	void LibLinearTrainer::InnerTrain(Instances& instances)
	{
		prob = Instances2Problem(instances);

		const char *error_msg = check_parameter(&prob, &param);
		if (error_msg)
		{
			fprintf(stderr, "ERROR: %s\n", error_msg);
			exit(1);
		}

		{
			Notifer timer("LibLinear train");
			model_ = train(&prob, &param);
		}
	}
	
	void LibLinearTrainer::Finalize_(Instances& instances)
	{
		_bias = model_->bias >= 0 ? model_->bias : 0;
	
		_weights.resize(_numFeatures, 0);
		double *w = model_->w;
		//int nr_class = model_->nr_class;
		//int nr_w;
		//if (nr_class == 2 && model_->param.solver_type != MCSVM_CS)
		//	nr_w = 1;
		//else
		//	nr_w = nr_class;

		//当前只考虑二分类 @TODO 多分类 	_weights[i] = w[i * nr_w]; linear.cpp 
		for (int i = 0; i < _numFeatures; i++)
		{
			_weights[i] = w[i]; 
		}

		free_and_destroy_model(&model_);
		destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
	}

}  //----end of namespace gezi

#endif  //----end of LIB_LINEAR_TRAINER_CPP_
