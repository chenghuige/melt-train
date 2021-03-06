/**
 *  ==============================================================================
 *
 *          \file   RegressionTree.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-08 19:22:50.386968
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef REGRESSION_TREE_H_
#define REGRESSION_TREE_H_

#include "Trainers/Gbdt/OnlineRegressionTree.h"
#include "Dataset.h"
#include "Feature.h"
#include "rabit_util.h"
DECLARE_int32(distributeMode); //@TODO
namespace gezi {

  //Not is a..., here because online will use a partiaion of regression tree, so..
  class RegressionTree : public OnlineRegressionTree
  {
  protected:
    vector<uint> _threshold; //online是Float离线训练其实是uint 覆盖掉基类中的_threashold
    const Feature* _features = NULL;
  public:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar & CEREAL_BASE_OBJECT_NVP(OnlineRegressionTree);
      ar & CEREAL_NVP(_threshold);
      ar & CEREAL_NVP(_features);
    }

    RegressionTree(int maxLeaves, const vector<Feature>& features)
    {
      _weight = 1.0;
      Reset(maxLeaves);
      NumLeaves = 1;
      _features = gezi::begin_ptr(features);
    }

    void Reset(int maxLeaves)
    {
      _splitFeature.resize(maxLeaves - 1);
      _splitGain.resize(maxLeaves - 1);
      _gainPValue.resize(maxLeaves - 1);
      _previousLeafValue.resize(maxLeaves - 1);
      _threshold.resize(maxLeaves - 1); //online是Float离线训练其实是uint 
      OnlineRegressionTree::_threshold.resize(_threshold.size());
      _lteChild.resize(maxLeaves - 1);
      _gtChild.resize(maxLeaves - 1);
      _leafValue.resize(maxLeaves);
      _parent.resize(maxLeaves);
      _depth.resize(maxLeaves, 0);
    }

    void Print()
    {
      Pvector(_splitFeature);
      Pvector(_splitGain);
      Pvector(_gainPValue);
      Pvector(_lteChild);
      Pvector(_gtChild);
      Pvector(_threshold);
      Pvector(_leafValue);
      Pval(NumLeaves);
    }

    void Print(const vector<Feature>& features)
    {
      Pvector(_splitFeature);
      Pvector(_splitGain);
      //Pvector(_gainPValue);
      Pvector(_lteChild);
      Pvector(_gtChild);
      //	Pval(NumLeaves);
      Fvec threshold(_threshold.size());
      for (size_t i = 0; i < _threshold.size(); i++)
      {
        threshold.push_back(features[_splitFeature[i]].BinUpperBounds[_threshold[i]]);
      }
      //	Pvector(_threshold);
      Pvector(threshold);
      int NumInternalNodes = NumLeaves - 1;
      Pval(NumInternalNodes);
      Pvector(_leafValue);
    }

    using OnlineRegressionTree::Print;
    void Print(const FeatureBin& features, int node = 0, int depth = 0, string suffix = "$")
    {
      Print(features, RegressionTree::_threshold, node, depth, suffix);
    }
    //void Finalize()
    //{
    //	//Reset(NumLeaves); //@TODO do not need this?
    //}

    //转换放到每一次split的过程中 该函数废弃
    void ToOnline(const vector<Feature>& features)
    {
      for (size_t i = 0; i < _threshold.size(); i++)
      {
        OnlineRegressionTree::_threshold[i] = features[_splitFeature[i]].BinUpperBounds[_threshold[i]];
      }
    }

    //score已经resize好
    template<typename T>
    void AddOutputsToScores(const T& dataset, Fvec& scores)
    {
#pragma omp parallel for
      for (size_t d = 0; d < dataset.size(); d++)
      {
        scores[d] += GetOutput(dataset[d]);
      }
    }

    template<typename T>
    void AddOutputsToScores(const T& dataset, Fvec& scores, Float multiplier)
    {
#pragma omp parallel for
      for (size_t d = 0; d < dataset.size(); d++)
      {
        scores[d] += multiplier * GetOutput(dataset[d]);
      }
    }

    template<typename T>
    void AddOutputsToScores(const T& dataset, Fvec& scores, const ivec& docIndices)
    {
#pragma omp parallel for
      for (size_t d = 0; d < docIndices.size(); d++)
      {
        scores[docIndices[d]] += GetOutput(dataset[docIndices[d]]);
      }
    }

    void ScaleOutputsBy(Float scalar)
    {
      for (auto& val : _leafValue)
      {
        val *= scalar;
      }
    }

    template<typename T>
    Float GetOutput(const T& featureBin)
    {
      if (_lteChild[0] == 0)
      { //training may has this case? @TODO
        return 0.0;
      }
      int leaf = GetLeaf(featureBin);
      return GetOutput(leaf);
    }

    int GetLeaf(const FeatureBin& featureBin)
    {
      return GetLeaf_(featureBin, _threshold);
    }

    int GetLeaf(const InstancePtr& instance)
    {
      return GetLeaf_(*instance, OnlineRegressionTree::_threshold);
    }

    //@TODO range ? for IEnumerable ?
    ivec GetNodesLeaves(int node)
    {
      if (NumLeaves == 1)
      {
        return ivec({ 0 });
      }
      if (node < 0)
      {
        return ivec({ ~node });
      }

      //@TODO try cpplinq concatenate
      ivec left = GetNodesLeaves(_lteChild[node]);
      ivec right = GetNodesLeaves(_gtChild[node]);
      left.insert(left.end(), right.begin(), right.end());
      return left;
    }

    Float GetOutput(int leaf)
    {
      return _leafValue[leaf];
    }

    template<typename T>
    Fvec GetOutputs(const T& dataset)
    {
      Fvec outputs(dataset.size());
#pragma omp parallel for
      for (size_t d = 0; d < dataset.size(); d++)
      {
        outputs[d] = GetOutput(dataset[d]);
      }
      return outputs;
    }

    int GTChild(int node)
    {
      return _gtChild[node];
    }

    Float LeafValue(int leaf)
    {
      return _leafValue[leaf];
    }

    int LTEChild(int node)
    {
      return _lteChild[node];
    }

    int Split(int leaf, int feature, uint threshold, Float LTEValue, Float GTValue, Float gain, Float gainPValue)
    {
      int indexOfNewInternal = NumLeaves - 1;
      /*	int parent = find_index(_lteChild, ~leaf);
        if (parent < _lteChild.size())
        {
        _lteChild[parent] = indexOfNewInternal;
        }
        else
        {
        parent = find_index(_gtChild, ~leaf);
        if (parent < _gtChild.size())
        {
        _gtChild[parent] = indexOfNewInternal;
        }
        }*/
      if (NumLeaves > 1)
      {
        int parent = _parent[leaf];
        if (parent >= 0)
        {
          _lteChild[parent] = indexOfNewInternal;
        }
        else
        {
          _gtChild[~parent] = indexOfNewInternal;
        }
      }
      _splitFeature[indexOfNewInternal] = feature;
      _splitGain[indexOfNewInternal] = gain;
      _gainPValue[indexOfNewInternal] = gainPValue;
      _threshold[indexOfNewInternal] = threshold;
      //每轮都立即处理而不是最后统一处理主要为了eval的需要
      //OnlineRegressionTree::_threshold[indexOfNewInternal] = _features[feature].BinUpperBounds[threshold];
      if (Rabit::GetWorldSize() == 1 || FLAGS_distributeMode != 1)
      {
        OnlineRegressionTree::_threshold[indexOfNewInternal] = _features[feature].BinUpperBounds[threshold];
      }
      else
      {
        gezi::Notifer notifer("Broadcast upperThreShold", 2);
        OnlineRegressionTree::_threshold[indexOfNewInternal] = Rabit::Choose(feature) ? _features[feature].BinUpperBounds[threshold] : 0;
        Rabit::Broadcast(OnlineRegressionTree::_threshold[indexOfNewInternal], feature % Rabit::GetWorldSize());
      }
      _lteChild[indexOfNewInternal] = ~leaf;
      _previousLeafValue[indexOfNewInternal] = _leafValue[leaf];
      _leafValue[leaf] = LTEValue;
      _parent[leaf] = indexOfNewInternal;
      _gtChild[indexOfNewInternal] = ~NumLeaves;
      _leafValue[NumLeaves] = GTValue;
      _parent[NumLeaves] = ~indexOfNewInternal;

      _depth[leaf] += 1;
      _depth[NumLeaves] = _depth[leaf];
      if (_depth[leaf] > _maxDepth)
      {
        _maxDepth = _depth[leaf];
      }

      if (LTEValue > _maxOutput)
      {
        _maxOutput = LTEValue;
      }
      if (GTValue > _maxOutput)
      {
        _maxOutput = GTValue;
      }
      NumLeaves++;
      return indexOfNewInternal;
    }

    int SplitFeature(int node)
    {
      return _splitFeature[node];
    }

    void SetOutput(int leaf, Float value)
    {
      _leafValue[leaf] = value;
    }

    const int NumNodes() const
    {
      return NumLeaves - 1;
    }

    int Depth(int node = 0) const 
    {
      if (node >= 0)
      {
        return 1 + std::max(Depth(_lteChild[node]), Depth(_gtChild[node]));
      }
      else
      {
        return 0;
      }
    }

    int MaxDepth() const
    {
      return _maxDepth;
    }
    
    int GetDepth(int node) const
    {
      return _depth[node];
    }

    const ivec& GetDepthVec() const
    {
      return _depth;
    }

    void UpdateOutputWithDelta(int leafIndex, double delta)
    {
      _leafValue[leafIndex] += delta;
    }

  private:
    ivec _parent; //每个叶子节点对应的parent index
    ivec _depth; //每个叶子节点对应的树的高度
    int _maxDepth = 0;
  };

}  //----end of namespace gezi

#endif  //----end of REGRESSION_TREE_H_
