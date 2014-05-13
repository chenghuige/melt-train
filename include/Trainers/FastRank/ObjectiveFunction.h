/**
 *  ==============================================================================
 *
 *          \file   ObjectiveFunction.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-10 22:37:44.351396
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef OBJECTIVE_FUNCTION_H_
#define OBJECTIVE_FUNCTION_H_
#include "Dataset.h"
#include "random_util.h"
#include "common_util.h"
namespace gezi {

	class ObjectiveFunction
	{
	public:
		Dataset& Dataset;
		dvec& Weights()
		{
			return _weights;
		}
	protected:
		bool _bestStepRankingRegressionTrees = false;
		dvec _gradient;
		int _gradSamplingRate;
		double _learningRate;
		double _maxTreeOutput = std::numeric_limits<double>::max();
		static const int _queryThreadChunkSize = 100;
		Random _rnd;
		dvec _weights;

	public:
		ObjectiveFunction(gezi::Dataset& dataset, double learningRate, double maxTreeOutput, int gradSamplingRate, bool useBestStepRankingRegressionTree, int randomNumberGeneratorSeed)
			:Dataset(dataset), _rnd(randomNumberGeneratorSeed)
		{
			_learningRate = learningRate;
			_maxTreeOutput = maxTreeOutput;
			_gradSamplingRate = gradSamplingRate;
			_bestStepRankingRegressionTrees = useBestStepRankingRegressionTree;
			_gradient.resize(Dataset.NumDocs);
			_weights.resize(Dataset.NumDocs);
		}

		virtual dvec& GetGradient(const dvec& scores)
		{
			int sampleIndex = _rnd.Next(_gradSamplingRate);
			//@TODO原代码采用100个分组 每个线程处理100个query
#pragma omp parallel for
			for (int query = 0; query < Dataset.NumDocs; query++)
			{
				if ((query % _gradSamplingRate) == sampleIndex)
				{ //@TODO 测试一下 inline 普通 虚函数的时间花费 是否改为类似Normalizer使用的function设计性能会好？
					GetGradientInOneQuery(query, scores);
				}
			}
			/*	Pvector(scores)
				Pvector(_gradient)*/
			return _gradient;
		}

	protected:
		//virtual void GetGradientInOneQuery(int query, const dvec& scores) = 0;
		std::function<void(int, const dvec&)> GetGradientInOneQuery;
		//这个去掉虚函数之后 速度能从6.32726 s ->6.19447 s 考虑虚函数之外的设计 特别对于这种内部嵌入的循环内核心虚函数 尽量采用function替代
	};

	typedef shared_ptr<ObjectiveFunction> ObjectiveFunctionPtr;
}  //----end of namespace gezi

#endif  //----end of OBJECTIVE_FUNCTION_H_
