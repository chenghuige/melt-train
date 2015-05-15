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
		Fvec& Weights()
		{
			return _weights;
		}
	protected:
		bool _bestStepRankingRegressionTrees = false;
		Fvec _gradient;
		int _gradSamplingRate;
		Float _learningRate;
		Float _maxTreeOutput = std::numeric_limits<Float>::max();
		static const int _queryThreadChunkSize = 100;
		Random _rnd;
		Fvec _weights;

	public:
		ObjectiveFunction(gezi::Dataset& dataset, Float learningRate, Float maxTreeOutput, int gradSamplingRate, bool useBestStepRankingRegressionTree, int randomNumberGeneratorSeed)
			:Dataset(dataset), _rnd(randomNumberGeneratorSeed)
		{
			_learningRate = learningRate;
			_maxTreeOutput = maxTreeOutput;
			_gradSamplingRate = gradSamplingRate;
			_bestStepRankingRegressionTrees = useBestStepRankingRegressionTree;
			_gradient.resize(Dataset.NumDocs);
			_weights.resize(Dataset.NumDocs);
		}
		
		//@TODO原代码采用100个分组 每个线程处理100个query`
		virtual Fvec& GetGradient(const Fvec& scores)
		{
			int sampleIndex = _rnd.Next(_gradSamplingRate);
#pragma omp parallel for
			for (int query = 0; query < Dataset.NumDocs; query++)
			{
				if ((query % _gradSamplingRate) == sampleIndex)
				{ 
					GetGradientInOneQuery(query, scores);
				}
			}
			/*	Pvector(scores)
				Pvector(_gradient)*/
			return _gradient;
		}

	protected:
		//virtual void GetGradientInOneQuery(int query, const Fvec& scores) = 0;
		//这个去掉虚函数之后 速度能从6.32726 s ->6.19447 s 考虑虚函数之外的设计 特别对于这种内部嵌入的循环内核心虚函数 尽量采用function替代
		std::function<void(int, const Fvec&)> GetGradientInOneQuery;
	};

	typedef shared_ptr<ObjectiveFunction> ObjectiveFunctionPtr;
}  //----end of namespace gezi

#endif  //----end of OBJECTIVE_FUNCTION_H_
