/**
 *  ==============================================================================
 *
 *          \file   Trainers/FastRank/Feature.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-07 16:10:12.796777
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__FAST_RANK__FEATURE_H_
#define TRAINERS__FAST_RANK__FEATURE_H_
#include "common_util.h"
#include "Numeric/Vector/TVector.h"
namespace gezi {

	class Feature
	{
	public:
		Feature() = default;
		virtual ~Feature() {}
		Feature(Feature&&) = default;
		Feature& operator = (Feature&&) = default;
		Feature(const Feature&) = default;
		Feature& operator = (const Feature&) = default;

		Feature(string name_, Fvec& binUpperBounds_, Fvec& binMedians_, IntArray& bins_)
			:Name(name_), BinUpperBounds(binUpperBounds_), BinMedians(binMedians_), Bins(bins_)
		{

		}

		void Set(string name_, Fvec& binUpperBounds_, Fvec& binMedians_, IntArray& bins_)
		{
			Name = name_;
			BinUpperBounds = binUpperBounds_;
			BinMedians = binMedians_;
			Bins = bins_;
		}

		int NumBins()
		{
			return BinUpperBounds.size();
		}
	public:
		string Name;
		Fvec BinUpperBounds;
		Fvec BinMedians;
		IntArray Bins;
		double Trust = 1;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__FEATURE_H_
