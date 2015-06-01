/**
 *  ==============================================================================
 *
 *          \file   Trainers/EnsembleTrainer.h
 *
 *        \author   chenghuige
 *
 *          \date   2015-04-16 20:03:19.082694
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef TRAINERS__ENSEMBLE_TRAINER_H_
#define TRAINERS__ENSEMBLE_TRAINER_H_

#include "MLCore/Trainer.h"
#include "MLCore/TrainerFactory.h"
#include "Predictors/EnsemblePredictor.h"
#include "Prediction/Instances/instances_util.h"
DECLARE_uint64(rs);
namespace gezi {

	//当前先只考虑多个相同类型的分类器训练，主要问题解析命令行的设计 都是按照运行时候只有一个tranier设计 多个可能冲突 @TODO
	class EnsembleTrainer : public ValidatingTrainer
	{
	public:
		EnsembleTrainer() = default;
		virtual PredictorPtr CreatePredictor() override
		{
			return make_shared<EnsemblePredictor>(move(_predictors), _calibrator);
		}

		virtual void ParseArgs() override;

		//@TODO 采样需要别的方法吗 比如不放回随机的随机分给几个trainer 感觉没啥用
		virtual void Train(Instances& instances)  override
		{
                        VLOG(5) << "Ensemble train"; 
			Init();
                        Random rand(_randSeed);
			for (size_t i = 0; i < _trainers.size(); i++)
			{
                                VLOG(2) << "Ensemble train with trainer " << i;
				Instances partionInstances = InstancesUtil::GenPartionInstances(instances, rand, _sampleRate);
				_trainers[i]->Train(partionInstances); //@TODO 也可以考虑内部trainer也validate Train(instance, {}, {}),似乎意义不大
                                _predictors.push_back(_trainers[i]->CreatePredictor());
                                ValidatingTrainer::SetScale(i + 1);
                                ValidatingTrainer::Evaluate(i + 1);
			}
			Finalize(instances);
		}

		virtual void GenPredicts() override
                {
                    DoAddPredicts([&](InstancePtr instance) {
                            return _predictors.back()->Output(instance);
                        });
                }

		virtual void Finalize(Instances& instances) override
		{
			if (_calibrator != nullptr)
			{
				VLOG(0) << "Calirate once for the final ensemble model";
                                if (!_selfEvaluate)
                                {
                                    VLOG(0) << "Normal calibrate for all instances";
				    _calibrator->Train(instances, [this](InstancePtr instance) {
					return Margin(instance); },
                                        _maxCalibrationExamples); 
			        }
                                else 
                                {
                                    VLOG(0) << "Quick calibrate since has done self evaluate";
                                    for (auto& item : Scores.back())
                                    {
                                        item /= (double)_predictors.size();    
                                    }
                                    _calibrator->Train(Scores.back(), instances);    
                                }
		        } 
                
                }

	protected:
		virtual void Init()
		{
			ParseArgs();
			for (size_t i = 0; i < _trainerNames.size(); i++)
			{
				for (int j = 0; j < _numTrainers[i]; j++)
				{
                                        FLAGS_rs += 1024; //确保每隔一trainer的起始randseed不同
					_trainers.push_back(TrainerFactory::CreateTrainer(_trainerNames[i]));
				}
			}
		}

		virtual Float Margin(InstancePtr instance)
		{
			double output = 0;
//#pragma omp parallel for reduction(+: output)
			for (size_t i = 0; i < _predictors.size(); i++)
			{
				output += _predictors[i]->Output(instance);
			}
			return output / _predictors.size();
		}
	protected:
		vector<string> _trainerNames = { "gbdt" };  //有几种不同类型的trainer
		vector<int> _numTrainers = { 5 }; //每种trainer的数目
		vector<TrainerPtr> _trainers;

		double _sampleRate = 0.7;
		size_t _maxCalibrationExamples = 1000000;
                unsigned _randSeed = 0x7b;

		vector<PredictorPtr> _predictors;
	private:
    };

}  //----end of namespace gezi

#endif  //----end of TRAINERS__ENSEMBLE_TRAINER_H_
