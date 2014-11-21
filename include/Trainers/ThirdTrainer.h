/**
 *  ==============================================================================
 *
 *          \file   Trainers/ThirdTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-20 16:33:18.028897
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__THIRD_TRAINER_H_
#define TRAINERS__THIRD_TRAINER_H_

#include "MLCore/Trainer.h"
namespace gezi {

	class ThirdTrainer : public Trainer
	{
	public:
		virtual void ParseArgs() override;
		virtual void Init() override;
	protected:
		string _classiferSettings;
		unsigned _randSeed = 0;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__THIRD_TRAINER_H_
