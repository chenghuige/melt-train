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

namespace gezi {

	class ScoreTracker
	{
	public:
		Dataset& Dataset;
		string DatasetName;
		dvec Scores;
	protected:
		dvec& _initScores;

	public: 
		ScoreTracker(ScoreTracker&&) = default;
		ScoreTracker& operator = (ScoreTracker&&) = default;
		ScoreTracker(const ScoreTracker&) = default;
		ScoreTracker& operator = (const ScoreTracker&) = default;

		ScoreTracker(string datasetName, gezi::Dataset& set, dvec& initScores)
			:Dataset(set), DatasetName(datasetName), _initScores(initScores)
		{
			InitScores(initScores);
		}


		virtual void AddScores(RegressionTree& tree, double multiplier)
		{
			tree.AddOutputsToScores(Dataset, Scores, multiplier);
			SendScoresUpdatedMessage();
		}

		virtual void AddScores(RegressionTree& tree, DocumentPartitioning& partitioning, double multiplier)
		{
			for (int l = 0; l < tree.NumLeaves; l++)
			{
				int begin;
				int count;
				ivec& documents = partitioning.ReferenceLeafDocuments(l, begin, count);
				double output = tree.LeafValue(l) * multiplier;
				int end = begin + count;
#pragma  omp parallel for
				for (int i = begin; i < end; i++)
				{
					Scores[documents[i]] += output;
				}
				SendScoresUpdatedMessage();
			}
		}

		virtual void InitScores(dvec& initScores)
		{
			if (initScores.empty())
			{
				if (Scores.empty())
				{
					Scores.resize(Dataset.NumDocs);
				}
				else
				{
					zeroset(Scores);
				}
			}
			else
			{
				if (initScores.size() != Dataset.NumDocs)
				{
					THROW("The length of initScores do not match the length of training set");
				}
				LOG(INFO) << "init scores with initScores" << initScores.size();
				Scores = initScores;
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

		virtual void SetScores(dvec& scores)
		{
			Scores.swap(scores);
			SendScoresUpdatedMessage();
		}

	};

	typedef shared_ptr<ScoreTracker> ScoreTrackerPtr;
}  //----end of namespace gezi

#endif  //----end of SCORE_TRACKER_H_
