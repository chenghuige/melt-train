/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An auc Op.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using std::vector;
//@TODO add weight as optional input
REGISTER_OP("Auc")
.Input("predicts: T1")
.Input("labels: T2")
.Output("z: float")
.Attr("T1: {float, double}")
.Attr("T2: {float, double}")
//.Attr("T1: {float, double}")
//.Attr("T2: {int32, int64}")
.SetIsCommutative()
.Doc(R"doc(
Given preidicts and labels output it's auc
)doc");

class AucOp : public OpKernel {
public:
	explicit AucOp(OpKernelConstruction* context) : OpKernel(context) {}

	template<typename ValueVec>
	void index_sort(const ValueVec& valueVec, vector<int>& indexVec)
	{
		indexVec.resize(valueVec.size());
		for (size_t i = 0; i < indexVec.size(); i++)
		{
			indexVec[i] = i;
		}
		std::sort(indexVec.begin(), indexVec.end(),
			[&valueVec](const int l, const int r) { return valueVec(l) > valueVec(r); });
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& predicts_tensor = context->input(0);
		const Tensor& labels_tensor = context->input(1);
		auto predicts = predicts_tensor.flat<float>(); //输入能接受float double那么这里如何都处理?
		auto labels = labels_tensor.flat<float>();

		vector<int> indexes;
		index_sort(predicts, indexes);
		
		typedef float Float;

		Float oldFalsePos = 0;
		Float oldTruePos = 0;
		Float falsePos = 0;
		Float truePos = 0;
		Float oldOut = std::numeric_limits<Float>::infinity();
		Float result = 0;

		for (size_t i = 0; i < indexes.size(); i++)
		{
			int index = indexes[i];
			Float label = labels(index);
			Float prediction = predicts(index);
			Float weight = 1.0;
			//Pval3(label, output, weight);
			if (prediction != oldOut) //存在相同值得情况是特殊处理的
			{
				result += 0.5 * (oldTruePos + truePos) * (falsePos - oldFalsePos);
				oldOut = prediction;
				oldFalsePos = falsePos;
				oldTruePos = truePos;
			}
			if (label > 0)
				truePos += weight;
			else
				falsePos += weight;
		}
		result += 0.5 * (oldTruePos + truePos) * (falsePos - oldFalsePos);
		Float AUC = result / (truePos * falsePos);

		// Create an output tensor
		Tensor* output_tensor = NULL;
		TensorShape output_shape;

		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
		
		output_tensor->scalar<float>()() = AUC;
	}
};

REGISTER_KERNEL_BUILDER(Name("Auc").Device(DEVICE_CPU), AucOp);
