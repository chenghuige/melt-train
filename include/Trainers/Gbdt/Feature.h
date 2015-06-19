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
#include "rabit_util.h"
namespace gezi {

	//class Feature : public RabitObject
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

		int NumBins() const
		{
			return BinUpperBounds.size();
		}

		friend class cereal::access;
		template<class Archive>
		void serialize(Archive &ar, const unsigned int version)
		{
			ar & Name;
			ar & BinUpperBounds;
			ar & BinMedians;
			ar & Bins;
			ar & Trust;
		}

	public:
		string Name;
		Fvec BinUpperBounds; //@TODO 如果多个DataSet的话 需要共享BinUpperBounds和BinMedians 需要shared_ptr 当然拷贝代价也不大。。
		Fvec BinMedians;
		IntArray Bins;
		Float Trust = 1;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__FAST_RANK__FEATURE_H_
