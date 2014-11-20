/**
 *  ==============================================================================
 *
 *          \file   ThirdTrainers.cpp
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 06:01:08.123837
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef THIRD_TRAINERS_CPP_
#define THIRD_TRAINERS_CPP_

#include "Trainers/SofiaTrainer.h"

DECLARE_string(cls);
DECLARE_uint64(rs);
namespace gezi {

	//@TODO 所有第三方的共同特性抽出
	void SofiaTrainer::ParseArgs()
	{
		PVAL(FLAGS_cls);
		if (!FLAGS_cls.empty())
		{
			_classiferSettings = gezi::replace_chars(FLAGS_cls, "=,", ' ');
		}
		_randSeed = FLAGS_rs;
	}

}  //----end of namespace gezi

#endif  //----end of THIRD_TRAINERS_CPP_
