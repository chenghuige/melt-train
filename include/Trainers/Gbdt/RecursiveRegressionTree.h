/**
 *  ==============================================================================
 *
 *          \file   RecursiveRegressionTree.h
 *
 *        \author   chenghuige
 *
 *          \date   2016-07-05 22:21:34.858339
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef GEZI_RECURSIVE_REGRESSION_TREE_H_
#define GEZI_RECURSIVE_REGRESSION_TREE_H_

#include "RegressionTreeNodeDocuments.h"
#include "RegressionTree.h"
#include "DocumentPartitioning.h"
namespace gezi {

	class RecursiveRegressionTree : RegressionTreeNodeDocuments
	{
	private:
		int _nodeCount;
		double _weightedOutput;
		shared_ptr<RecursiveRegressionTree> GTNode;
		shared_ptr<RecursiveRegressionTree> LTENode;
	public:
		RecursiveRegressionTree(RegressionTree& t, const DocumentPartitioning& p, int n)
			:RegressionTreeNodeDocuments(t, p, n)
		{
			_weightedOutput = std::numeric_limits<double>::quiet_NaN();
			_nodeCount = 0x7fffffff;
			if (!IsLeaf())
			{
				LTENode = make_shared<RecursiveRegressionTree>(Tree, Partitioning, Tree.LTEChild(NodeIndex));
				GTNode = make_shared<RecursiveRegressionTree>(Tree, Partitioning, Tree.GTChild(NodeIndex));
			}
		}

		double GetWeightedOutput()
		{
			//if (_weightedOutput == std::numeric_limits<double>::quiet_NaN())  //@WARNING this is now work!
			if (std::isnan(_weightedOutput))
			{
				if (NodeIndex < 0)
				{
					return Tree.GetOutput(~NodeIndex);
				}
				int lteDocCount = LTENode->GetDocumentCount();
				int gtCount = GTNode->GetDocumentCount();
				//Pval2(lteDocCount, gtCount);
				_weightedOutput = ((lteDocCount * LTENode->GetWeightedOutput()) + (gtCount * GTNode->GetWeightedOutput())) / ((double)(lteDocCount + gtCount));
			}
			//Pval(_weightedOutput);
			return _weightedOutput;
		}
	
		void SmoothLeafOutputs(double parentOutput, double smoothing)
		{
			double myOutput = ((1.0 - smoothing) * GetWeightedOutput()) + (smoothing * parentOutput);
			if (IsLeaf())
			{
				//Pval2(Tree.GetOutput(~NodeIndex), myOutput);
				Tree.SetOutput(~NodeIndex, myOutput);
			}
			else
			{
				LTENode->SmoothLeafOutputs(myOutput, smoothing);
				GTNode->SmoothLeafOutputs(myOutput, smoothing);
			}
		}

	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of GEZI_RECURSIVE_REGRESSION_TREE_H_
