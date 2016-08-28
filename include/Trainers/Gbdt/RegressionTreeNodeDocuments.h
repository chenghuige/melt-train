/**
 *  ==============================================================================
 *
 *          \file   RegressionTreeNodeDocuments.h
 *
 *        \author   chenghuige
 *
 *          \date   2016-07-05 22:21:48.819404
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef GEZI_REGRESSION_TREE_NODE_DOCUMENTS_H_
#define GEZI_REGRESSION_TREE_NODE_DOCUMENTS_H_

#include "RegressionTree.h"
#include "DocumentPartitioning.h"

namespace gezi {

	class RegressionTreeNodeDocuments
	{
	private:
		int documentCount = -1;
	public:
		int NodeIndex;
		const DocumentPartitioning& Partitioning;
		RegressionTree& Tree;
	public:
		RegressionTreeNodeDocuments(RegressionTree& tree, const DocumentPartitioning& partitioning, int nodeIndex)
			:Tree(tree), Partitioning(partitioning),
			NodeIndex(nodeIndex)
		{
		}

		int GetDocumentCount()
		{
			if (documentCount == -1)
			{
				documentCount = from(Tree.GetNodesLeaves(NodeIndex))
					>> select([this](int leaf) {return Partitioning.NumDocsInLeaf(leaf); })
					>> cpplinq::sum();
			}
			return documentCount;
		}

		bool IsLeaf()
		{
			return NodeIndex < 0;
		}

		//not used
		void UpdateOutputsWithDelta(double delta)
		{
			if (IsLeaf())
			{
				Tree.UpdateOutputWithDelta(~NodeIndex, delta);
			}
			else
			{
				for (int leaf : Tree.GetNodesLeaves(NodeIndex))
				{
					Tree.UpdateOutputWithDelta(leaf, delta);
				}
			}
		}

	protected:
	private:

	};

}  //----end of namespace gezi

#endif  //----end of GEZI_REGRESSION_TREE_NODE_DOCUMENTS_H_
