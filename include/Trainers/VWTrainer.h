/**
 *  ==============================================================================
 *
 *          \file   Trainers/VWTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-11-19 15:40:10.886164
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__V_W_TRAINER_H_
#define TRAINERS__V_W_TRAINER_H_

#include "MLCore/Trainer.h"
#include "ThirdTrainer.h"
#include "MLCore/Predictor.h"

class example;
class vw;
namespace VW{
	class primitive_feature_space;
}
namespace gezi {

	//暂时不支持VW解析输入 @TODO
	class VWTrainer : public ThirdTrainer
	{
	public:
		virtual PredictorPtr CreatePredictor() override;

	protected:
		example* Instance2Example(InstancePtr instance, bool includeLabel);

		virtual void Initialize(Instances& instances) override;

		virtual void InnerTrain(Instances& instances) override;

		virtual Float Margin(InstancePtr instance) override;
		virtual void Finalize_(Instances& instances) override;

	public:
		static VW::primitive_feature_space* pFeatureSpace();
		VW::primitive_feature_space* _pFeatureSpace;
		vw* _vw = NULL;
		string modelFile;
		string readableModelFile;
	};

}  //----end of namespace gezi

#endif  //----end of TRAINERS__V_W_TRAINER_H_
