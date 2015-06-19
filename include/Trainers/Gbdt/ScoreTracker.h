/**
 *  ==============================================================================
 *
 *          \file   ScoreTracker.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-10 17:37:55.051231
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef SCORE_TRACKER_H_
#define SCORE_TRACKER_H_

#include "common_util.h"
#include "Dataset.h"
#include "Prediction/Instances/Instances.h"

namespace gezi {

	class ScoreTracker
	{
	public:
		const Dataset* Dataset = NULL; //ֻ��TrainSet ��ת������DataSetʹ�õ�
		const Instances* Instances = NULL; //validating ��ʱ��ʹ�� 
		string DatasetName;
		Fvec& Scores;
	protected:
		int _numDocs;

	public:
		ScoreTracker(ScoreTracker&&) = default;
		ScoreTracker& operator = (ScoreTracker&&) = default;
		ScoreTracker(const ScoreTracker&) = default;
		ScoreTracker& operator = (const ScoreTracker&) = default;

		ScoreTracker(string datasetName, const gezi::Dataset& set, Fvec& initScores)
			:Dataset(&set), DatasetName(datasetName), Scores(initScores), _numDocs(set.size())
		{
			InitScores();
		}

		ScoreTracker(string datasetName, const gezi::Instances& instances, Fvec& initScores)
			:Instances(&instances), DatasetName(datasetName), Scores(initScores), _numDocs(instances.size())
		{
			InitScores();
		}

		virtual void AddScores(RegressionTree& tree, Float multiplier)
		{
			if (Dataset)
			{
				tree.AddOutputsToScores(*Dataset, Scores, multiplier);
			}
			if (Instances)
			{
				tree.AddOutputsToScores(*Instances, Scores, multiplier);
			}
			SendScoresUpdatedMessage();
		}

		virtual void AddScores(RegressionTree& tree, DocumentPartitioning& partitioning, Float multiplier)
		{
			for (int l = 0; l < tree.NumLeaves; l++)
			{
				int begin;
				int count;
				ivec& documents = partitioning.ReferenceLeafDocuments(l, begin, count);
				Float output = tree.LeafValue(l) * multiplier;
				int end = begin + count;
#pragma  omp parallel for
				for (int i = begin; i < end; i++)
				{
					Scores[documents[i]] += output;
				}
				SendScoresUpdatedMessage();
			}
		}

		virtual void InitScores()
		{
			if (Scores.size() != _numDocs)
			{
				//Scores.resize(_numDocs, 0); //ע��resize�����󵼡��� ���ǵ��ڵ�����С Ȼ������
				gezi::reset_vec(Scores, _numDocs, 0);
			}
			SendScoresUpdatedMessage();
		}

		void RandomizeScores(int rngSeed, bool reverseRandomization)
		{
			Random rndStart(rngSeed);
			for (size_t i = 0; i < Scores.size(); i++)
			{
				Scores[i] += (10.0 * rndStart.NextDouble()) * (reverseRandomization ? -1.0 : 1.0);
			}
			SendScoresUpdatedMessage();
		}

		//@TODO
		void SendScoresUpdatedMessage()
		{

		}

		virtual void SetScores(Fvec& scores)
		{
			Scores.swap(scores);
			SendScoresUpdatedMessage();
		}

	};

	typedef shared_ptr<ScoreTracker> ScoreTrackerPtr;

	//--�����������Ƹ����� ���Ǻ����������̳�ScoreTracker��Ƚ��鷳 ����  RankScoreTracker ��ô�ֵ���DataSetRank..,InsatncesRank..  OverDesign
	//class DatSetScoreTracker : public ScoreTracker
	//{
	//public:
	//	ScoreTracker(string datasetName, const gezi::Dataset& set, Fvec& initScores)
	//		:Dataset(&set), DatasetName(datasetName), Scores(initScores), _numDocs(set.size())
	//	{
	//		InitScores();
	//	}
	//};

	//class InstancesScoreTracker : public ScoreTracker
	//{
	//public:
	//	ScoreTracker(string datasetName, const gezi::Dataset& set, Fvec& initScores)
	//		:Dataset(&set), DatasetName(datasetName), Scores(initScores), _numDocs(set.size())
	//	{
	//		InitScores();
	//	}
	//};
}  //----end of namespace gezi

#endif  //----end of SCORE_TRACKER_H_
