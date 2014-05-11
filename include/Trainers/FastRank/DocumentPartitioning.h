/**
 *  ==============================================================================
 *
 *          \file   DocumentPartitioning.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 15:31:14.841064
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef DOCUMENT_PARTITIONING_H_
#define DOCUMENT_PARTITIONING_H_

#include "common_util.h"
#include "RegressionTree.h"
#include "Dataset.h"
namespace gezi {

	class DocumentPartitioning
	{
	public:
		DocumentPartitioning(RegressionTree tree, Dataset dataset)
			: DocumentPartitioning(dataset.NumDocs, tree.NumLeaves)
		{
			vector<ivec> perLeafDocumentLists(tree.NumLeaves);
#pragma omp parallel for
			for (int d = 0; d < dataset.NumDocs; d++)
			{
				int leaf = tree.GetLeaf(dataset.GetFeatureBinRow(d));
#pragma omp critical
				{
					perLeafDocumentLists[leaf].push_back(d);
				}
			}

			for (int i = 0; i < tree.NumLeaves; i++)
			{
				_leafCount[i] = perLeafDocumentLists[i].size();
				_leafBegin[i] = i == 0 ? 0 : _leafBegin[i - 1] + _leafCount[i];
			}

#pragma omp parallel for
			for (int leaf = 0; leaf < tree.NumLeaves; leaf++)
			{
				int documentPos = _leafBegin[leaf];
				for (int d : perLeafDocumentLists[leaf])
				{
					_documents[documentPos++] = d;
				}
			}
		}

		DocumentPartitioning(int numDocuments, int maxLeaves)
		{
			_leafBegin.resize(maxLeaves);
			_leafCount.resize(maxLeaves);
			_documents.resize(numDocuments);
		}

		DocumentPartitioning(ivec& documents, int numDocuments, int maxLeaves)
			: DocumentPartitioning(numDocuments, maxLeaves)
		{
			//@TODO will below ok?
			//_initialDocuments.assign(documents.begin(), documents.end());
			_initialDocuments.resize(numDocuments);
			for (int d = 0; d < numDocuments; d++)
			{
				_initialDocuments[d] = documents[d];
			}
		}

		void Initialize()
		{
			zeroset(_leafCount);
			_leafBegin[0] = 0;
			_leafCount[0] = _documents.size();
			if (_initialDocuments.empty())
			{
				for (int d = 0; d < _documents.size(); d++)
				{
					_documents[d] = d;
				}
			}
			else
			{
				for (int d = 0; d < _documents.size(); d++)
				{
					_documents[d] = _initialDocuments[d];
				}
			}
		}

		int GetLeafDocuments(int leaf, ivec& documents)
		{
			fill_range(documents.begin(), _documents.begin() + _leafBegin[leaf], _leafCount[leaf]);
			return _leafCount[leaf];
		}

		void Split(int leaf, IntArray& indexer, uint threshold, int gtChildIndex)
		{
			if (_tempDocuments.empty())
			{
				_tempDocuments.resize(_documents.size());
			}
			int begin = _leafBegin[leaf];
			int end = begin + _leafCount[leaf];
			int newEnd = begin;
			int tempEnd = begin;
			for (int curr = begin; curr < end; curr++)
			{
				int doc = _documents[curr];
				if (indexer[doc] > threshold)
				{
					_tempDocuments[tempEnd++] = doc;
				}
				else
				{
					_documents[newEnd++] = doc;
				}
			}
			int newCount = newEnd - begin;
			int gtCount = tempEnd - begin;
			fill_range(_documents.begin() + newEnd, _tempDocuments.begin() + begin, gtCount);
			_leafCount[leaf] = newCount;
			_leafBegin[gtChildIndex] = newEnd;
			_leafCount[gtChildIndex] = gtCount;
		}

		double Mean(dvec& array, int leaf, bool filterZeros)
		{
			double mean = 0.0;
			int end = _leafBegin[leaf] + _leafCount[leaf];
			int count = filterZeros ? 0 : _leafCount[leaf];
			if (filterZeros)
			{
				for (int i = _leafBegin[leaf]; i < end; i++)
				{
					double value = array[_documents[i]];
					if (value != 0.0)
					{
						mean += value;
						count++;
					}
				}
			}
			else
			{
				for (int i = _leafBegin[leaf]; i < end; i++)
				{
					mean += array[_documents[i]];
				}
			}
			return (mean / ((double)count));
		}

		double Mean(dvec& array, dvec& sampleWeights, int leaf, bool filterZeros)
		{
			if (sampleWeights.empty())
			{
				return Mean(array, leaf, filterZeros);
			}
			double mean = 0.0;
			int end = _leafBegin[leaf] + _leafCount[leaf];
			double sumWeight = 0.0;
			if (filterZeros)
			{
				for (int i = _leafBegin[leaf]; i < end; i++)
				{
					double value = array[_documents[i]];
					if (value != 0.0)
					{
						double weight = sampleWeights[_documents[i]];
						mean += value * weight;
						sumWeight += weight;
					}
				}
			}
			else
			{
				for (int i = _leafBegin[leaf]; i < end; i++)
				{
					double weight = sampleWeights[_documents[i]];
					mean += array[_documents[i]] * weight;
					sumWeight += weight;
				}
			}
			return (mean / sumWeight);
		}

		void ReferenceLeafDocuments(int leaf, ivec* pdocuments, int& begin, int& count)
		{
			pdocuments = &_documents;
			begin = _leafBegin[leaf];
			count = _leafCount[leaf];
		}

		int NumDocs()
		{
			return _documents.size();
		}

		int NumDocsInLeaf(int leaf)
		{
			return _leafCount[leaf];
		}

	protected:
	private:
		ivec _documents; //doc id����Ҷ��˳��0-NumLeaves�������� leaf�ڲ���doc id�ɷ����� Ӧ�ÿ���?@TODO
		ivec _initialDocuments; //��ʼdoc id����
		ivec _leafBegin;  //ÿ��Ҷ�Ӷ�Ӧdoc ��Ŀ�ۼ� ���� 3 4 5 -> 3 7 12
		ivec _leafCount; //ÿ��Ҷ�Ӷ�Ӧ��doc��Ŀ
		ivec _tempDocuments;
	};

}  //----end of namespace gezi

#endif  //----end of DOCUMENT_PARTITIONING_H_