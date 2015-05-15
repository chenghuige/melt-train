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
		
		//@TODOԭ�������100������ ÿ���̴߳���100��query`
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
		//���ȥ���麯��֮�� �ٶ��ܴ�6.32726 s ->6.19447 s �����麯��֮������ �ر���������ڲ�Ƕ���ѭ���ں����麯�� ��������function���
		std::function<void(int, const Fvec&)> GetGradientInOneQuery;
	};

	typedef shared_ptr<ObjectiveFunction> ObjectiveFunctionPtr;
}  //----end of namespace gezi

#endif  //----end of OBJECTIVE_FUNCTION_H_
