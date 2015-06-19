/**
 *  ==============================================================================
 *
 *          \file   OptimizationAlgorithm.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 14:35:32.002966
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef OPTIMIZATION_ALGORITHM_H_
#define OPTIMIZATION_ALGORITHM_H_
#include "common_def.h"
#include "TreeLearner.h"
#include "RegressionTree.h"
#include "ObjectiveFunction.h"
#include "ScoreTracker.h"
#include "Ensemble.h"
#include "IStepSearch.h"
#include "Prediction/Instances/Instances.h"
namespace gezi {

	class OptimizationAlgorithm
	{
	public:
		OptimizationAlgorithm(gezi::Ensemble& ensemble, const Dataset& trainData, Fvec& initTrainScores)
			:Ensemble(ensemble)
		{

		}

		OptimizationAlgorithm(gezi::Ensemble& ensemble)
			:Ensemble(ensemble)
		{

		}

		//��Ϊc++�Ĺ��캯���в������麯�� ConstructScoreTracker ������� ��c#��һ��  @TODO check����gcc֧�ֹ��캯�����麯������c#��
		virtual void Initialize(const Dataset& trainData, Fvec& initTrainScores)
		{
			TrainingScores = ConstructScoreTracker("itrain", trainData, initTrainScores);
			TrackedScores.push_back(TrainingScores);
		}

		virtual RegressionTree& TrainingIteration(const BitArray& activeFeatures) = 0;
		virtual ScoreTrackerPtr ConstructScoreTracker(string name, const Dataset& set, Fvec& initScores)
		{
			return make_shared<ScoreTracker>(name, set, initScores);
		}
		virtual ScoreTrackerPtr ConstructScoreTracker(string name, const Instances& instances, Fvec& initScores)
		{
			return make_shared<ScoreTracker>(name, instances, initScores);
		}

		virtual void FinalizeLearning(int bestIteration)
		{
			if (bestIteration != Ensemble.NumTrees())
			{
				Ensemble.RemoveAfter(std::max(bestIteration, 0));
				TrackedScores.clear();
			}
		}

		//��ǰ�����һ���� ���bagging nbag > 1��ô �����¶�ÿ��test set������tracker �����������õ�score���� ��������Ч��һ�� �����ۻ�score
		//������Χ�ֶ�moveһ�� TrackedScores,�����������弸��û��
		ScoreTrackerPtr GetScoreTracker(string name, const Instances& set, Fvec& InitScores)
		{
			for (ScoreTrackerPtr st : TrackedScores)
			{
				if (st->DatasetName == name)
				{
					return st;
				}
			}
			ScoreTrackerPtr newTracker = ConstructScoreTracker(name, set, InitScores);
			TrackedScores.push_back(newTracker);
			return newTracker;
		}

		void SetTrainingData(const Dataset& trainData, Fvec& initTrainScores)
		{
			TrainingScores = ConstructScoreTracker("itrain", trainData, initTrainScores);
			TrackedScores[0] = TrainingScores;
		}

		virtual void SmoothTree(RegressionTree& tree, Float smoothing)
		{
			AutoTimer timer("SmoothTree");
			//if (smoothing != 0.0)
			{//@TODO smooth

			}
		}

		virtual void UpdateAllScores(RegressionTree& tree)
		{
			//dynamic_pointer_cast<IStepSearch>(ObjectiveFunction))->AdjustTreeOutputsAutoTimer timer("UpdateAllScores");
			for (ScoreTrackerPtr t : TrackedScores)
			{
				UpdateScores(t, tree);
			}
		}

		virtual void UpdateScores(ScoreTrackerPtr t, RegressionTree& tree)
		{
			if (t == TrainingScores)
			{
				if (AdjustTreeOutputsOverride != nullptr)
				{ //@TODO
					//VLOG(2) << "AdjustTreeOutputsOverride != nullptr";
				}
				else
				{
					//VLOG(2) << "t->AddScores(tree, TreeLearner->Partitioning, 1.0)";
					t->AddScores(tree, TreeLearner->Partitioning, 1.0);
				}
			}
			else
			{
				//VLOG(2) << "t->AddScores(tree,1.0)";
				t->AddScores(tree, 1.0);
			}
		}

	public:
		Ensemble& Ensemble;
		IStepSearchPtr AdjustTreeOutputsOverride = nullptr;
		TreeLearnerPtr TreeLearner = nullptr;
		ObjectiveFunctionPtr ObjectiveFunction = nullptr;
		Float Smoothing = 0.0;
		vector<ScoreTrackerPtr> TrackedScores;
		ScoreTrackerPtr TrainingScores = nullptr;
	protected:
		bool _useFastTrainingScoresUpdate = true;
	};

	typedef shared_ptr<OptimizationAlgorithm> OptimizationAlgorithmPtr;
}  //----end of namespace gezi

#endif  //----end of OPTIMIZATION_ALGORITHM_H_