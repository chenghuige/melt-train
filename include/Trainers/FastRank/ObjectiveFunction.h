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
		ObjectiveFunction(::Dataset& dataset, double learningRate, double maxTreeOutput, int gradSamplingRate, bool useBestStepRankingRegressionTree, int randomNumberGeneratorSeed)
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
			//@TODOԭ�������100������ ÿ���̴߳���100��query
#pragma omp parallel for
			for (int q = 0; q < Dataset.NumDocs; q++)
			{
				if ((q % _gradSamplingRate) == sampleIndex)
				{ //@TODO ����һ�� inline ��ͨ �麯����ʱ�仨�� �Ƿ��Ϊ����Normalizerʹ�õ�function������ܻ�ã�
					GetGradientInOneQuery(q, scores);
				}
			}
			return _gradient;
		}

	protected:
		virtual void GetGradientInOneQuery(int query, const dvec& scores) = 0;
	};

	typedef shared_ptr<ObjectiveFunction> ObjectiveFunctionPtr;
}  //----end of namespace gezi

#endif  //----end of OBJECTIVE_FUNCTION_H_
