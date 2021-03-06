/**
*  ==============================================================================
*
*          \file   Prediction/Instances/InstanceParser.h
*
*        \author   chenghuige
*
*          \date   2014-03-27 17:47:57.438203
*
*  \Description:   输入是数据文件 可能是dense 或者 sparse格式 输出是Instances
*                  @TODO目前只考虑Instances内容数组都保存在内存 不考虑 StreamingInstances @TODO
* 输入数据格式 dense (default)
*  //with header
*	#id label age weight
*  _123 0    32   18
*  //whithout header
*  _123 0  32 18 //id label feature1 feature2
* 输入数据格式 sparse (default)
*   0 5 0:0.4 3:0.1
*   1 5 1:0.3
* 内部数据处理和TLC一致 不管输入是dense还是sparse都是判断当前0的数目 如果> 1/2 * FeatureNum 则dense否则sparse
@TODO should be InsancesParser
*  ==============================================================================
*/

#ifndef PREDICTION__INSTANCES__INSTANCE_PARSER_H_
#define PREDICTION__INSTANCES__INSTANCE_PARSER_H_
#define  NO_BAIDU_DEP
#include "common_util.h"
#include "RegexSearcher.h"
#include "Identifer.h"
#include "Prediction/Instances/Instances.h"

#include "rabit_util.h"

#include "serialize_util.h"

namespace gezi {

  //@TODO 修改名字或者增加namespace 易冲突
  static const map<string, FileFormat> kFormats = {
    { "unknown", FileFormat::Unknown },
    { "dense", FileFormat::Dense },
    { "tsv", FileFormat::Tsv },
    { "sparse", FileFormat::Sparse },
    { "sparse_no_length", FileFormat::SparseNoLength },
    { "sparse2", FileFormat::SparseNoLength },
    { "text", FileFormat::Text },
    { "libsvm", FileFormat::LibSVM },
    { "arff", FileFormat::Arff },
    { "vw", FileFormat::VW },
    { "malloc", FileFormat::MallocRank },
    { "mallocrank", FileFormat::MallocRank },
  };

  //RunConvert 显示转换使用
  static const map<FileFormat, string> kFormatNames = {
    { FileFormat::Unknown, "unknown" },
    { FileFormat::Dense, "dense" },
    { FileFormat::Tsv, "tsv" },
    { FileFormat::Sparse, "sparse" },
    { FileFormat::SparseNoLength, "sparse" },
    { FileFormat::Text, "txt" },
    { FileFormat::LibSVM, "libsvm" },
    { FileFormat::Arff, "arff" },
    { FileFormat::VW, "vw" },
    { FileFormat::MallocRank, "malloc" },
  };

  //其它最后输出 使用
  static const map<FileFormat, string> kFormatSuffixes = {
    { FileFormat::Unknown, "txt" },
    { FileFormat::Dense, "txt" },
    { FileFormat::Sparse, "txt" },
    { FileFormat::SparseNoLength, "sparse" },
    { FileFormat::Text, "txt" },
    { FileFormat::LibSVM, "libsvm" },
    { FileFormat::Arff, "arff" },
    { FileFormat::VW, "vw" },
    { FileFormat::MallocRank, "malloc" },
  };


  //不做clear清理工作每次 create 一个新的InstanceParser使用即可
  class InstanceParser
  {
  public:
    struct Arguments
    {
      int labelIdx = -1;
      int weightIdx = -1;
      //nameIdx|, seprated like 1,2,3, name filed will be shown in .inst.txt result file
      string namesIdx = "";
      //|the same as nameIdx, attrIdx will be filed to be ignored
      string attrsIdx = "";
      string groupsIdx = "";
      bool hasHeader = false; //|header: no header by default
      string sep = "tab"; //|or space or something like , ; etc. 
      string ncsep = "|"; //|contact names filed like pid|title|content 4003|good title|good content
      //|excl vs. incl determines whether features for the expression below are included or excluded. expression=[s:substringList | r:regexList | i:indexList | e:exactNameList]
      string excl = "";
      //|use excl will exlude those specified, use incl will only include those specified, use incl + excl means first incl then excl
      string incl = "";
      bool keepSparse = false; //sparse|
      bool keepDense = false; //dense|
      string inputFormat = "normal";//format|support melt/tlc format as normal, also support libSVM, may support weka/arff, malloc format later
      int libsvmStartIndex = 1;
      double sparsifyThre = 0.5;
      string resultDir = "";//rd|
      bool cacheInstance = false;//|if cacheInsantce will seralize instance as binary
      string featureNameFile;
    };

    InstanceParser()
    {
      ParseArguments();
      InitParam();
    }

  public:
    Instances&& Parse(string dataFile, bool printInfo = false)
    {
      Noticer timer("ParseInputDataFile", 0);
      _instances.name = dataFile;
      Parse_(dataFile);
      Finallize();

      if (printInfo)
      {
        PrintInfo();
      }
      return move(_instances);
    }

    //默认输入是dense数据，并且没有header,解析后 添加fake header 到outFile
    void AddFakeHeader(string dataFile, string outFile)
    {
      //@TODO streaming ?
      vector<string> lines = read_lines_fast(dataFile, "//");
      if (lines.empty())
      {
        LOG(FATAL) << "Fail to load data file! " << dataFile << " is empty!";
      }

      //判断schema信息
      ParseFirstLine(lines);


      ofstream ofs(outFile);

      //-----------------写fake header
      size_t featureIdx = 0, nameIdx = 0, attributeIdx = 0;
      ofs << "#";
      switch (_columnTypes[0])
      {
      case ColumnType::Feature:
        ofs << format("f{}", featureIdx++);
        break;
      case ColumnType::Name:
        ofs << format("name{}", nameIdx++);
        break;
      case ColumnType::Label:
        ofs << "label";
        break;
      case ColumnType::Weight:
        ofs << "weight";
        break;
      case ColumnType::Attribute:
        ofs << format("attr{}", attributeIdx++);
        break;
      default:
        break;
      }
      for (size_t i = 1; i < _columnTypes.size(); i++)
      {
        switch (_columnTypes[i])
        {
        case ColumnType::Feature:
          ofs << "\t" << format("f{}", featureIdx++);;
          break;
        case ColumnType::Name:
          ofs << "\t" << format("name{}", nameIdx++);
          break;
        case ColumnType::Label:
          ofs << "\t" << "label";
          break;
        case ColumnType::Weight:
          ofs << "\t" << "weight";
          break;
        case ColumnType::Attribute:
          ofs << "\t" << format("attr{}", attributeIdx++);
          break;
        default:
          break;
        }
      }
      ofs << endl;

      for (string line : lines)
      {
        ofs << line << endl;
      }
    }

  public:
    void ParseArguments();

    Arguments& Args()
    {
      return _args;
    }

    ivec GetIndexesFromInput(string input)
    {
      ivec result;
      svec sv = split(input, ",");
      for (auto item : sv)
      {
        if (contains(item, "-"))
        {
          svec t = split(item, "-");
          int start = INT(t[0]);
          int end = INT(t[1]) + 1;
          for (int i = start; i < end; i++)
          {
            result.push_back(i);
          }
        }
        else
        {
          result.push_back(INT(item));
        }
      }
      return result;
    }

    void InitParam()
    {
      _format = boost::to_lower_copy(_args.inputFormat);
      if (_args.sep == "tab")
      {
        _sep = "\t";
      }
      else if (_args.sep == "space")
      {
        _sep = " ";
      }
      else if (_args.sep.empty())
      {
        _sep = "\t";
      }
      else
      {
        _sep = _args.sep;
      }

      if (_format == "libsvm")
      {
        _sep = " ";
      }


      if (!_args.attrsIdx.empty())
      {
        _attributesIdx = GetIndexesFromInput(_args.attrsIdx);
        PVEC(_attributesIdx);
      }

      if (!_args.namesIdx.empty())
      {
        _namesIdx = GetIndexesFromInput(_args.namesIdx);
        PVEC(_namesIdx);
      }

      if (!_args.groupsIdx.empty())
      {
        _groupsIdx = GetIndexesFromInput(_args.groupsIdx);
        PVEC(_groupsIdx);
      }
    }

    //@TODO 当前值考虑支持一种 比如 i, e, s, r不考虑多个同时存在,一般 最常用的r 正则匹配即可
    ivec GetSelectedFeatures(string input)
    {
      string pattern, content;
      if (contains(input, ":"))
      {
        pattern = input.substr(0, 1);
        content = input.substr(2);
      }
      else
      {
        pattern = "r";
        content = input;
      }
      if (pattern == "i")
      {
        return GetIndexesFromInput(content);
      }
      ivec result;
      svec t = split(content, ",");
      set<string> s(t.begin(), t.end());
      if (pattern == "e")
      { //exact
        for (int i = 0; i < _numFeatures; i++)
        {
          if (s.count(_instances.schema.featureNames[i]))
          {
            result.push_back(i);
          }
        }
      }
      else if (pattern == "s")
      { //substr
        for (int i = 0; i < _numFeatures; i++)
        {
          for (auto p : s)
          {
            if (contains(_instances.schema.featureNames[i], p))
            {
              result.push_back(i);
            }
          }
        }
      }
      else if (pattern == "r")
      { //regex
        RegexSearcher rs;
        rs.add(s);
        for (int i = 0; i < _numFeatures; i++)
        {
          if (rs.has_match(_instances.schema.featureNames[i]))
          {
            result.push_back(i);
          }
        }
      }
      else
      {
        LOG(FATAL) << "Unsupported pattern " << input << " only support i: e: s: r:";
      }
      return result;
    }

    BitArray GetSelectedArray()
    {
      BitArray filterArray;
      if (!_args.incl.empty())
      {
        ivec incls = GetSelectedFeatures(_args.incl);

        if (incls.empty())
        {
          LOG(WARNING) << "Do not has any match for incl " << _args.incl;
        }
        else
        {
          filterArray.resize(_numFeatures, false);

          if (incls.size() > 100)
          {
            VLOG(0) << "Total incls features is " << incls.size() << " only print top 100";
          }
          else
          {
            VLOG(0) << "Total incls features is " << incls.size();
          }

          int count = 0;
          for (int idx : incls)
          {
            count++;
            if (count <= 100)
            {
              VLOG(0) << "Including feature: " << _instances.schema.featureNames[idx];
            }
            filterArray[idx] = true;
          }
        }
      }

      if (!_args.excl.empty())
      {
        ivec excls = GetSelectedFeatures(_args.excl);

        if (excls.empty())
        {
          LOG(WARNING) << "Do not has any match for excl " << _args.excl;
        }
        else
        {
          if (filterArray.empty())
          {
            filterArray.resize(_numFeatures, true);
          }

          if (excls.size() > 100)
          {
            VLOG(0) << "Total excls features is " << excls.size() << " only print top 100";
          }
          else
          {
            VLOG(0) << "Total excls features is " << excls.size();
          }

          int count = 0;
          for (int idx : excls)
          {
            count++;
            if (count <= 100)
            {
              VLOG(0) << "Excluding feature: " << _instances.schema.featureNames[idx];
            }

            filterArray[idx] = false;
          }
        }
      }

      if (filterArray.empty())
      {
        filterArray.resize(_numFeatures, true);
      }

      return filterArray;
    }

    bool IsSparse() const
    {
      return !IsDense();
    }

    bool IsDense() const
    {
      return _fileFormat == FileFormat::Dense;
    }

    bool IsDenseFormat() const
    {
      return _fileFormat == FileFormat::Dense;
    }

    void InitColumnTypes(const svec& lines)
    {
      int maxNameAtrrIdx = -1;
      for (int idx : _attributesIdx)
      {
        if (idx > maxNameAtrrIdx)
        {
          maxNameAtrrIdx = idx;
        }
        _columnTypes[idx] = ColumnType::Attribute;
      }


      for (int idx : _namesIdx)
      {
        if (idx > maxNameAtrrIdx)
        {
          maxNameAtrrIdx = idx;
        }
        _columnTypes[idx] = ColumnType::Name;
      }

      for (int idx : _groupsIdx)
      {
        _groupKeysMark[idx] = true;
      }

      if (_args.weightIdx >= 0)
      {
        _columnTypes[_args.weightIdx] = ColumnType::Weight;
        _hasWeight = true;
      }

      //--------------set label
      if (_args.labelIdx >= 0)
      {
        _labelIdx = _args.labelIdx;
      }

      if (startswith(_firstColums[0], '#'))
      {
        _hasHeader = true;
        //_columnTypes[0] = ColumnType::Name;

        //if (_labelIdx < 0)
        //	{
        //	_labelIdx = 1;
        //	_columnTypes[1] = ColumnType::Label;
        //	}

        //if (_labelIdx < 0)
        //{
        //	_labelIdx = 0;
        //	_columnTypes[0] = ColumnType::Label;
        //}
      }

      //前面确认了是否有header
      if (_hasHeader)
      {
        _headerColums.swap(_firstColums);
        _firstColums = split(lines[_hasHeader], _sep);
      }

      //如果是_开头那么第一个是Name 如果没有设置labelIdx那么第二个紧跟着认为是Label
      if (startswith(_firstColums[0], '_'))
      {
        {
          _columnTypes[0] = ColumnType::Name;
          if (_labelIdx < 0)
          {
            if (maxNameAtrrIdx >= 0)
            {
              _labelIdx = maxNameAtrrIdx + 1;
            }
            else
            {
              _labelIdx = 1;
            }
          }
        }
      }

      //默认如果没有制定labelIdx 第一个是Label, 如果有name,attr的制定 那么name,attr后面的第一列是Label
      if (_labelIdx < 0)
      {//如果 maxNameAtrrIdx == -1 , then 0
        _labelIdx = maxNameAtrrIdx + 1;
      }

      Pval(_labelIdx);
      _columnTypes[_labelIdx] = ColumnType::Label;
    }

    //实际上Dense,Sparse格式 在这里完成了设置, SparseNoLength, Text将会在create instances中设置
    void InitNames()
    {
      if (_hasHeader)
      { //意味着文件中明确指明了所有的特征名称
        for (int i = 0; i < _columnNum; i++)
        {
          switch (_columnTypes[i])
          {
          case ColumnType::Feature:
            _instances.schema.featureNames.push_back(_headerColums[i]);
            break;
          case ColumnType::Attribute:
            _instances.schema.attributeNames.push_back(_headerColums[i]);
            break;
          case ColumnType::Name:
            _instances.schema.tagNames.push_back(_headerColums[i]);
            break;
          default:
            break;
          }
          if (_groupKeysMark[i])
          {
            _instances.schema.groupKeys.push_back(_headerColums[i]);
          }
        }
        _numFeatures = _instances.NumFeatures();
      }
      else
      { //需要统计特征数目
        int numFeatures = 0;
        if (_fileFormat == FileFormat::Dense)
        {
          for (auto type : _columnTypes)
          {
            if (type == ColumnType::Feature)
            {
              //string name = format("f{}", i);
              //_instances.schema.featureNames.emplace_back(name);
              numFeatures++;
            }
          }
          _numFeatures = numFeatures;
        }
        //else if (_fileFormat == FileFormat::Sparse)
        //{ //注意已经解析首个数据行 获取到 特征数目了(_numFeatures) 但是注意 SparseNoLength, Text获取不到
        //_instances.schema.featureNames.reserve(_numFeatures);
        //for (int i = 0; i < _numFeatures; i++) //@TODO may exceed int capacity
        //{
        //	string name = format("f{}", i);
        //	_instances.schema.featureNames.push_back(name);
        //}
        //}

        {//try to load from default feature name file
          VLOG(0) << "Try load feature names from " << _args.featureNameFile;
          _instances.schema.featureNames.Load(_args.featureNameFile);
          _instances.schema.featureNames.SetNumFeatures(_numFeatures);
          Pval_1(_instances.schema.featureNames.NumFeatures());
          Pval_1(_instances.schema.featureNames.NumFeatureNames());
        }
        {
          for (auto index : _namesIdx)
          {
            _instances.schema.tagNames.push_back(STR(index));
          }
          for (auto index : _attributesIdx)
          {
            _instances.schema.attributeNames.push_back(STR(index));
          }
          for (auto index : _groupsIdx)
          {
            _instances.schema.groupKeys.push_back(STR(index));
          }
        }
      }
      PVEC(_instances.schema.tagNames);
      PVEC(_instances.schema.groupKeys);
    }

    void SetTextFeatureNames()
    {
      if (!_hasHeader)
      {
        _instances.schema.featureNames = GetIdentifer().keys();
      }
    }

    void CreateInstancesFromDenseFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromDenseFormat";
      uint64 end = start + _instanceNum;
#pragma omp parallel for 
      for (uint64 i = start; i < end; i++)
      {
        string line = lines[i];
        _instances[i - start] = make_shared<Instance>(_numFeatures);
        Instance& instance = *_instances[i - start];
        Vector& features = instance.features;
        features.PrepareDense();

        int featureIndex = 0;
        svec groupKeys;
        int count = _groupsIdx.empty() ?
          split_enumerate(line, _sep[0], [&, this](int index, int start, int len) {
          switch (_columnTypes[index])
          {
          case ColumnType::Feature:
            double value; //必须单独一行。。 否则crosses initialization
            value = _selectedArray[featureIndex++] ? atof(line.c_str() + start) : 0;
            features.Add(value);
            break;
          case ColumnType::Name:
            instance.names.push_back(line.substr(start, len));
            break;
          case ColumnType::Label:
            instance.label = atof(line.c_str() + start);
            break;
          case ColumnType::Weight:
            instance.weight = atof(line.c_str() + start);
            break;
          case ColumnType::Attribute:
            instance.attributes.push_back(line.substr(start, len));
            break;
          default:
            break;
          } }) :
        split_enumerate(line, _sep[0], [&, this](int index, int start, int len) {
          switch (_columnTypes[index])
          {
          case ColumnType::Feature:
            double value; //必须单独一行。。 否则crosses initialization
            value = _selectedArray[featureIndex++] ? atof(line.c_str() + start) : 0;
            features.Add(value);
            break;
          case ColumnType::Name:
            instance.names.push_back(line.substr(start, len));
            if (_groupKeysMark[index])
            {
              groupKeys.push_back(instance.names.back());
            }
            break;
          case ColumnType::Label:
            instance.label = atof(line.c_str() + start);
            break;
          case ColumnType::Weight:
            instance.weight = atof(line.c_str() + start);
            break;
          case ColumnType::Attribute:
            instance.attributes.push_back(line.substr(start, len));
            if (_groupKeysMark[index])
            {
              groupKeys.push_back(instance.attributes.back());
            }
            break;
          default:
            break;
          } });
        instance.groupKey = gezi::join(groupKeys, _args.ncsep);

        if (count != _columnNum)
        {
          LOG(WARNING) << "has bad line " << i << "count: " << count << " _columnNum: " << _columnNum;
          LOG(WARNING) << line;
          _instances[i - start] = nullptr;
        }

        //svec l = split(line, _sep);
        ////CHECK_EQ(l.size(), _columnNum) << "has bad line " << i; //不允许有坏数据行
        //if ((int)l.size() != _columnNum)
        //{
        //	LOG(WARNING) << "has bad line " << i;
        //	LOG(WARNING) << line;
        //	continue;
        //}

        //int fidx = 0;
        //double value = 0;
        //for (int j = 0; j < _columnNum; j++)
        //{
        //	string item = l[j];
        //	switch (_columnTypes[j])
        //	{
        //	case ColumnType::Feature:
        //		value = _selectedArray[fidx++] ? DOUBLE(item) : 0;
        //		features.Add(value);
        //		break;
        //	case ColumnType::Name:
        //		instance.names.push_back(item);
        //		break;
        //	case ColumnType::Label:
        //		instance.label = DOUBLE(item);
        //		break;
        //	case ColumnType::Weight:
        //		instance.weight = DOUBLE(item);
        //		break;
        //	case ColumnType::Attribute:
        //		instance.attributes.push_back(item);
        //		break;
        //	default:
        //		break;
        //	}
        //}
      }
      ufo::erase(_instances, nullptr);
      _instanceNum = (uint64)_instances.size();
    }

    //这个放到这里会影响找不到所有其它stl_util.h里面定义的split...
    ////@TODO 这个版本是 int + double 如果是需要其他类型的 fast atou快一些 但是运行有问题 暂不使用
    //template<typename FindFunc, typename UnfindFunc>
    //static void split(string input, const char sep, const char inSep, FindFunc findFunc, UnfindFunc unfindFunc)
    //{
    //	size_t pos = 0;
    //	size_t pos2 = input.find(sep);
    //	int i = 0;
    //	while (pos2 != string::npos)
    //	{
    //		int len = pos2 - pos;
    //		//why find_char so slow.... 
    //		//size_t inPos = find_char(input, inSep, pos, len);
    //		size_t inPos = input.find(inSep, pos);
    //		//if (inPos != string::npos)
    //		if (inPos < pos2) //这个查找可以优化 但是问题不大。。
    //		{
    //			findFunc(atoi(input.c_str() + pos), atof(input.c_str() + inPos + 1));
    //			//findFunc(fast_atou(input.c_str() + pos, input.c_str() + inPos), atof(input.c_str() + inPos + 1));
    //		}
    //		else
    //		{
    //			unfindFunc(i, input.substr(pos, len));
    //		}
    //		pos = pos2 + 1;
    //		pos2 = input.find(sep, pos);
    //		i++;
    //	}
    //	size_t inPos = input.find(inSep, pos);
    //	findFunc(atoi(input.c_str() + pos), atof(input.c_str() + inPos + 1));
    //}
    //@TODO Instance Next(string line) 支持streaming
    //@TODO tlc貌似快很多thread.feature.txt 9w instance 
    //1000ms 这里需要2000ms是因为split c#的速度更快？ 另外注意能char split find不要用string
    //不过使用omp并行后 200ms就搞定
    void CreateInstancesFromSparseFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << " CreateInstancesFromSparseFormat";
      uint64 end = start + _instanceNum;
#pragma omp parallel for 
      for (uint64 i = start; i < end; i++)
      {
        string line = lines[i];
        _instances[i - start] = make_shared<Instance>(_numFeatures);
        Instance& instance = *_instances[i - start];
        Vector& features = instance.features;
        if (_groupsIdx.empty())
        {
          splits_int_double(line, _sep[0], ':',
            [&, this](int index, Float value)
          {
            if (_selectedArray[index])
            {
              features.Add(index, value);
            }
          },
            [&, this](int index, string item)
          {
            ParseSparseAttributes(instance, index, item);
          });
        }
        else
        {
          svec groupKeys;
          splits_int_double(line, _sep[0], ':',
            [&, this](int index, Float value)
          {
            if (_selectedArray[index])
            {
              features.Add(index, value);
            }
          },
            [&, this](int index, string item)
          {
            ParseSparseAttributes(instance, index, item, groupKeys);
          });
          instance.groupKey = gezi::join(groupKeys, _args.ncsep);
        }
        //svec l = split(line, _sep);
        //for (size_t j = 0; j < l.size(); j++)
        //{
        //	string item = l[j];
        //	int index;
        //	Float value;
        //	bool ret = split(item, ':', index, value);
        //	if (ret)
        //	{
        //		if (_selectedArray[index])
        //			features.Add(index, value);
        //	}
        //	else
        //	{
        //		//@TODO 暂时没有检查格式正确性 要求的是 所有非:的数据 都在后面
        //		switch (_columnTypes[j])
        //		{
        //		case ColumnType::Name:
        //			instance.names.push_back(item);
        //			break;
        //		case ColumnType::Label:
        //			instance.label = DOUBLE(item);
        //			break;
        //		case ColumnType::Weight:
        //			instance.weight = DOUBLE(item);
        //			break;
        //		case ColumnType::Attribute:
        //			instance.attributes.push_back(item);
        //			break;
        //		default:
        //			break;
        //		}
        //	}
        //}
      }

      //int count = 0;
      //for (auto& instance : _instances)
      //{
      //	//cout << "al:" << instance->attributes.size() << endl;
      //	if (instance->attributes.size() == 0)
      //	{
      //		cout << lines[count] << endl;
      //	}
      //	count++;
      //}
    }

    //如果没有提前声明向量最大长度SparseNoLength 那么不支持特征选择,否则影响解析速度 可以先转换为正常的sparse格式
    void CreateInstancesFromSparseNoLengthFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromSparseNoLengthFormat";
      uint64 end = start + _instanceNum;
      int maxIndex = 0;
      //#pragma omp parallel for 
#pragma omp parallel for reduction(max : maxIndex)
      for (uint64 i = start; i < end; i++)
      {
        string line = lines[i];
        _instances[i - start] = make_shared<Instance>();
        Instance& instance = *_instances[i - start];
        Vector& features = instance.features;
        if (_groupsIdx.empty())
        {
          splits_int_double(line, _sep[0], ':', [&, this](int index, Float value) {
            if (_selectedArray[index])
            {
              features.Add(index, value);
            }

            if (index > maxIndex)
            {
              //#pragma omp critical //锁的代价 是否值得并行@TODO
              maxIndex = index;
            }
          },
            [&, this](int index, string item) {
            ParseSparseAttributes(instance, index, item);
          });
        }
        else
        {
          svec groupKeys;
          splits_int_double(line, _sep[0], ':', [&, this](int index, Float value) {
            if (_selectedArray[index])
            {
              features.Add(index, value);
            }

            if (index > maxIndex)
            {
              //#pragma omp critical //锁的代价 是否值得并行@TODO
              maxIndex = index;
            }
          },
            [&, this](int index, string item) {
            ParseSparseAttributes(instance, index, item, groupKeys);
          });
          instance.groupKey = gezi::join(groupKeys, _args.ncsep);
        }
        //				svec l = split(line, _sep);
        //
        //				for (size_t j = 0; j < l.size(); j++)
        //				{
        //					string item = l[j];
        //					string index_, value_;
        //					bool ret = split(item, ':', index_, value_);
        //					if (ret)
        //					{
        //						int index = INT(index_); Float value = DOUBLE(value_);
        //						if (_selectedArray[index])
        //							features.Add(index, value);
        //#pragma omp critical
        //						{
        //							if (index > maxIndex)
        //							{
        //								maxIndex = index;
        //							}
        //						}
        //					}
        //					else
        //					{
        //						//@TODO 暂时没有检查格式正确性 要求的是 所有非:的数据 都在后面 如果不符合会越界
        //						switch (_columnTypes[j])
        //						{
        //						case ColumnType::Name:
        //							instance.names.push_back(item);
        //							break;
        //						case ColumnType::Label:
        //							instance.label = DOUBLE(item);
        //							break;
        //						case ColumnType::Weight:
        //							instance.weight = DOUBLE(item);
        //							break;
        //						case ColumnType::Attribute:
        //							instance.attributes.push_back(item);
        //							break;
        //						default:
        //							break;
        //						}
        //					}
        //				}
      }
      _numFeatures = maxIndex + 1;
      Rabit::Allreduce<op::Max>(_numFeatures);
      _instances.schema.featureNames.SetNumFeatures(_numFeatures);
    }

    FileFormat GetFileFormat(string line)
    {
      if (_fileFormat != FileFormat::Unknown)
      {
        return _fileFormat;
      }

      for (int i = 0; i < _columnNum; i++)
      {
        if (_columnTypes[i] == ColumnType::Feature)
        {
          if (contains(_firstColums[i], ':'))
          {
            //return FileFormat::SparseNoLength; //去掉默认解析 sparse no length因为不常用,为了更好兼容libsvm格式
            return FileFormat::LibSVM;
          }

          if (i > 0 && contains(_firstColums[i], '|'))
          {
            return FileFormat::VW;
          }

          int j = i + 1;
          while (j < _columnNum && _columnTypes[j] != ColumnType::Feature)
          {
            j++;
          }

          if (j == _columnNum)
          {
            return FileFormat::Dense;
          }
          else
          {
            if (contains(_firstColums[j], ':'))
            {
              _numFeatures = INT(_firstColums[i]); //sparse格式解析出特征数目！ @IMPORTANT
              _columnTypes[i] = ColumnType::Attribute;
              return FileFormat::Sparse;
            }
            else
            {
              return FileFormat::Dense;
            }
          }
        }
      }
      LOG(FATAL) << "Data format wrong not feature field " << line;
    }

    void SetHeaderSchema(string line)
    {
      _instances.SetHeader(line, _hasHeader);
      _instances.schema.fileFormat = _fileFormat;
      _instances.schema.hasWeights = _hasWeight;
      _instances.schema.columnTypes = _columnTypes;
    }

    //获取列信息，名字，是dense还是sparse表示
    void ParseFirstLine(const svec& lines)
    {
      //Timer timer;
      string line = lines[0];
      _firstColums = split(line, _sep);

      if (_firstColums.size() < 2)
      {
        char sep = GuessSeparator(lines[0], "\t ,");
        _firstColums = split(line, sep);
        //_sep = string(sep); //貌似这样char to string 会core。。
        _sep = STRING(sep);
      }

      //VLOG(2) << format("split time: {}", timer.elapsed_ms());
      //timer.restart();

      _columnNum = _firstColums.size();
      PVAL(_columnNum);
      if (_columnNum < 2)
      {
        LOG(FATAL) << "The header must at least has label and one feature";
      }
      _columnTypes.resize(_columnNum, ColumnType::Feature);
      _groupKeysMark.resize(_columnNum, false);
      InitColumnTypes(lines);

      //VLOG(2) << format("InitColumnTypes time: {}", timer.elapsed_ms());
      //timer.restart();


      if (_fileFormat == FileFormat::Unknown)
      {
        _fileFormat = kFormats[_format];
        if (_fileFormat == FileFormat::Unknown)
        {
          _fileFormat = GetFileFormat(lines[_hasHeader]);
        }
      }

      InitNames();

      //VLOG(2) << format("InitNames time: {}", timer.elapsed_ms());
      //timer.restart();

      SetHeaderSchema(line);

      //VLOG(2) << format("SetHeaderSchema time: {}", timer.elapsed_ms());
      //timer.restart();
    }

    void PrintInfo()
    {
      Pval(kFormatNames[_fileFormat]);
      Pval(_instances.NumFeatures());
      Pval(_instances.Count());
      uint64 positiveCount = _instances.PositiveCount();
      Pval(positiveCount);
      Float positiveRatio = positiveCount / (double)_instances.Count();
      Pval(positiveRatio);

      //如果多个label打印每个label的instance数目信息
      if (_instances.schema.numClasses > 2)
      {
        int numLabels = _instances.schema.numClasses;
        Pval(numLabels);
        map<int, size_t> countMap;
        for (auto& instance : _instances)
        {
          countMap[(int)instance->label] += 1;
        }
        for (auto& item : countMap)
        {
          VLOG(0) << "Label:" << item.first << " Count:" << item.second;
        }
      }

      //如果是Ranking打印有多少组
      if (_instances.IsRankingInstances())
      {
        Pval(_instances.NumGroups());
      }

      uint64 denseCount = _instances.DenseCount();
      Float denseRatio = denseCount / (double)_instances.Count();
      Pval2(denseCount, denseRatio);
      Pval(IsDenseFormat());
      Pvec(_instances.schema.tagNames);
      Pvec(_instances.schema.attributeNames);
      Pvec(_instances.schema.groupKeys);
      Pvec_LastN(_instances.schema.featureNames, 10);
      Pval(_instances.schema.featureNames.NumFeatures());
      Pval(_instances.schema.featureNames.NumFeatureNames());
      Pval_1(_args.keepSparse);
      Pval_1(_args.keepDense);
    }

    char GuessSeparator(string line, string seps)
    {
      for (char sep : seps)
      {
        if (line.find(sep) != string::npos)
        {
          return sep;
        }
      }
      THROW(format("Could not gusess sep for line:[{}] seps:{}", line, seps));
    }

    void ParseSparseAttributes(Instance& instance, int index, string item)
    {
      switch (_columnTypes[index])
      {
      case ColumnType::Name:
        instance.names.push_back(item);
        break;
      case ColumnType::Label:
        instance.label = DOUBLE(item);
        /*				if (instance.label == -1)
        {
        instance.label = 0;
        }*/
        break;
      case ColumnType::Weight:
        instance.weight = DOUBLE(item);
        break;
      case ColumnType::Attribute:
        instance.attributes.push_back(item);
        break;
      default:
        break;
      }
    }

    //处理广义的属性数据
    void ParseSparseAttributes(Instance& instance, int index, string item, svec& groupKeys)
    {
      //switch (_columnTypes[index])
      //{
      //case ColumnType::Name:
      //	instance.names.push_back(item);
      //	break;
      //case ColumnType::Label:
      //	instance.label = DOUBLE(item);
      //	/*				if (instance.label == -1)
      //					{
      //					instance.label = 0;
      //					}*/
      //	break;
      //case ColumnType::Weight:
      //	instance.weight = DOUBLE(item);
      //	break;
      //case ColumnType::Attribute:
      //	instance.attributes.push_back(item);
      //	break;
      //default:
      //	break;
      //}
      ParseSparseAttributes(instance, index, item);
      if (_groupKeysMark[index])
      {
        groupKeys.push_back(item);
      }
    }


    //@TODO 还可以优化的是首先处理 不带有:的属性 然后集中处理带有的
    //@FIXM 目前只有label数据的 也就是空向量的会core
    void CreateInstancesFromLibSVMFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromLibSVMFormat";
      uint64 end = start + _instanceNum;
      int maxIndex = _args.libsvmStartIndex;
      //char sep = GuessSeparator(lines[0], "\t "); //已经在ParseFirstLine的时候确定了
      char sep = _sep[0];
#pragma omp parallel for reduction(max : maxIndex)
      for (uint64 i = start; i < end; i++)
      {
        string line = boost::trim_right_copy(lines[i]);
        _instances[i - start] = make_shared<Instance>(_numFeatures);
        Instance& instance = *_instances[i - start];
        Vector& features = instance.features;
        if (_groupsIdx.empty())
        {
          splits_int_double(line, sep, ':', [&, this](int index, Float value) {
            if (_selectedArray[index - _args.libsvmStartIndex])
            {
              features.Add(index - _args.libsvmStartIndex, value);
            }

            if (index > maxIndex)
            {
              maxIndex = index;
            }
          },
            [&, this](int index, string item) {
            ParseSparseAttributes(instance, index, item);
          });
        }
        else
        {
          svec groupKeys;
          splits_int_double(line, sep, ':', [&, this](int index, Float value) {
            if (_selectedArray[index - _args.libsvmStartIndex])
            {
              features.Add(index - _args.libsvmStartIndex, value);
            }

            if (index > maxIndex)
            {
              maxIndex = index;
            }
          },
            [&, this](int index, string item) {
            ParseSparseAttributes(instance, index, item, groupKeys);
          });
          instance.groupKey = gezi::join(groupKeys, _args.ncsep);
        }
        //				svec l = split(line, "\t "); //libsvm 是用空格或者tab都有可能
        //				for (size_t j = 0; j < l.size(); j++)
        //				{
        //					string item = l[j];
        //					if (j == 0)
        //					{
        //						instance.label = DOUBLE(item);
        //						if (instance.label == -1)
        //						{
        //							instance.label = 0;
        //						}
        //						continue;
        //					}
        //					string index_, value_;
        //					split(item, ':', index_, value_);
        //					int index = INT(index_); Float value = DOUBLE(value_);
        //#pragma omp critical
        //					{
        //						if (index > maxIndex)
        //						{
        //							maxIndex = index;
        //						}
        //					}
        //					features.Add(index - 1, value); //libsvm 是1开始 melt/tlc内部0开始处理
        //				}
      }
      _numFeatures = _args.libsvmStartIndex == 1 ? maxIndex : maxIndex + 1;
      Rabit::Allreduce<op::Max>(_numFeatures);
      _instances.schema.featureNames.SetNumFeatures(_numFeatures);
    }

    //针对malloc rank格式数据的 分割解析函数
    template<typename FindFunc, typename UnfindFunc>
    void splits_int_double_malloc(string input, const char sep, const char inSep, FindFunc findFunc, UnfindFunc unfindFunc)
    {
      size_t pos = 0;
      size_t pos2 = input.find(sep);
      int i = 0;
      bool isFirst = true;
      while (pos2 != string::npos)
      {
        int len = pos2 - pos;
        //why find_char so slow.... 
        //size_t inPos = find_char(input, inSep, pos, len);
        size_t inPos = input.find(inSep, pos);
        //if (inPos != string::npos)
        if (inPos < pos2) //这个查找可以优化 但是问题不大。。暂时没找到好的方法 改为find_char速度更慢很多
        {
          if (isFirst)
          {
            isFirst = false;
            unfindFunc(i, input.substr(pos, len));
          }
          else
          {
            //findFunc(atoi(input.c_str() + pos), atof(input.c_str() + inPos + 1));
            findFunc(fast_atou(input.c_str() + pos, input.c_str() + inPos), atof(input.c_str() + inPos + 1));
          }
        }
        else
        {
          unfindFunc(i, input.substr(pos, len));
        }
        pos = pos2 + 1;
        pos2 = input.find(sep, pos);
        i++;
      }
      size_t inPos = input.find(inSep, pos);
      findFunc(atoi(input.c_str() + pos), atof(input.c_str() + inPos + 1));
    }

    //类似libsvm 可能稠密或者稀疏 区别是 有一个 qid:0 类似的在feature前面
    void CreateInstancesFromMallocRankFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromMallocRankFormat";
      uint64 end = start + _instanceNum;
      int maxIndex = 1; //malloc 都是 下标1开始
      //char sep = GuessSeparator(lines[0], "\t "); //已经在ParseFirstLine的时候确定了
      char sep = _sep[0];
#pragma omp parallel for reduction(max : maxIndex)
      for (uint64 i = start; i < end; i++)
      {
        string line = boost::trim_right_copy(lines[i]);
        _instances[i - start] = make_shared<Instance>(_numFeatures);
        Instance& instance = *_instances[i - start];
        Vector& features = instance.features;

        svec groupKeys;
        splits_int_double_malloc(line, sep, ':', [&, this](int index, Float value) {
          if (_selectedArray[index - 1])
          {
            features.Add(index - 1, value);
          }

          if (index > maxIndex)
          {
            maxIndex = index;
          }
        },
          [&, this](int index, string item) {
          ParseSparseAttributes(instance, index, item, groupKeys);
        });
        instance.groupKey = gezi::join(groupKeys, _args.ncsep);
      }
      _numFeatures = maxIndex;
      Rabit::Allreduce<op::Max>(_numFeatures);
      _instances.schema.featureNames.SetNumFeatures(_numFeatures);
    }

    void CreateInstancesFromVWFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromVWFormat";
      uint64 end = start + _instanceNum;
      for (uint64 i = start; i < end; i++)
      {
        _instances[i - start] = make_shared<Instance>(_numFeatures);
        Instance& instance = *_instances[i - start];
        instance.line = lines[i];
        instance.label = lines[i][0] == '1' ? 1 : -1;
      }
    }

    //训练文本解析 暂时都是只支持单一char的分隔符 没有必要支持string分割
    void ParseTextForTrain(const svec& lines, uint64 start)
    {
      VLOG(0) << "ParseTextForTrain";
      uint64 end = start + _instanceNum;
#pragma omp parallel for 
      for (uint64 i = start; i < end; i++)
      {
        string line = lines[i];
        _instances[i - start] = make_shared<Instance>();
        Instance& instance = *_instances[i - start];
        Vector& features = instance.features;
        if (_groupsIdx.empty())
        {
          splits_string_double(line, _sep[0], ':', [&, this](string key, Float value) {
            int index;
#pragma  omp critical
            { //输入需要保证没有重复的key
              index = GetIdentifer().add(key);
            }
            if (_selectedArray[index])
            {
              features.Add(index, value);
            }
          },
            [&, this](int index, string item) {
            ParseSparseAttributes(instance, index, item);
          });
        }
        else
        {
          svec groupKeys;
          splits_string_double(line, _sep[0], ':', [&, this](string key, Float value) {
            int index;
#pragma  omp critical
            { //输入需要保证没有重复的key
              index = GetIdentifer().add(key);
            }
            if (_selectedArray[index])
            {
              features.Add(index, value);
            }
          },
            [&, this](int index, string item) {
            ParseSparseAttributes(instance, index, item, groupKeys);
          });
          instance.groupKey = gezi::join(groupKeys, _args.ncsep);
        }

        //				svec l = split(line, _sep);
        //
        //				for (size_t j = 0; j < l.size(); j++)
        //				{
        //					string item = l[j];
        //					string key, value_;
        //					int index; Float value;
        //					bool ret = split(item, ':', key, value_);
        //					if (ret)
        //					{
        //						value = DOUBLE(value_);
        //#pragma  omp critical
        //						{
        //							//输入需要保证没有重复的key
        //							index = GetIdentifer().add(key);
        //						}
        //						if (_selectedArray[index])
        //							features.Add(index, value);
        //					}
        //					else
        //					{
        //						//@TODO 暂时没有检查格式正确性 要求的是 所有非:的数据 都在后面
        //						switch (_columnTypes[j])
        //						{
        //						case ColumnType::Name:
        //							instance.names.push_back(item);
        //							break;
        //						case ColumnType::Label:
        //							instance.label = DOUBLE(item);
        //							break;
        //						case ColumnType::Weight:
        //							instance.weight = DOUBLE(item);
        //							break;
        //						case ColumnType::Attribute:
        //							instance.attributes.push_back(item);
        //							break;
        //						default:
        //							break;
        //						}
        //					}
        //				}
      }
      _numFeatures = GetIdentifer().size();
      SetTextFeatureNames();
      GetIdentifer().Save(_args.resultDir + "/identifer.bin");
    }

    //测试文本解析
    void ParseTextForTest(const svec& lines, uint64 start)
    {
      VLOG(0) << "ParseTextForTest";
      _numFeatures = GetIdentifer().size();
      if (_numFeatures == 0)
      { //可能RunTest模式 需要加载词表
        string path = _args.resultDir + "/identifer.bin";
        InstanceParser::GetIdentifer().Load(path);
        _numFeatures = GetIdentifer().size();
        CHECK(_numFeatures != 0) << "No identifer in memory or to load from disk " << path;
      }

      SetTextFeatureNames();
      uint64 end = start + _instanceNum;
#pragma omp parallel for 
      for (uint64 i = start; i < end; i++)
      {
        string line = lines[i];
        _instances[i - start] = make_shared<Instance>(_numFeatures);
        Instance& instance = *_instances[i - start];
        Vector& features = instance.features;
        if (_groupsIdx.empty())
        {
          splits_string_double(line, _sep[0], ':', [&, this](string key, Float value) {
            int index = GetIdentifer().id(key);
            if (index != Identifer::null_id() && _selectedArray[index])
            {
              features.Add(index, value);
            }
          },
            [&, this](int index, string item) {
            ParseSparseAttributes(instance, index, item);
          });
        }
        else
        {
          svec groupKeys;
          splits_string_double(line, _sep[0], ':', [&, this](string key, Float value) {
            int index = GetIdentifer().id(key);
            if (index != Identifer::null_id() && _selectedArray[index])
            {
              features.Add(index, value);
            }
          },
            [&, this](int index, string item) {
            ParseSparseAttributes(instance, index, item, groupKeys);
          });
          instance.groupKey = gezi::join(groupKeys, _args.ncsep);
        }

        //svec l = split(line, _sep);
        //for (size_t j = 0; j < l.size(); j++)
        //{
        //	string item = l[j];
        //	string key, value_;
        //	bool ret = split(item, ':', key, value_);
        //	if (ret)
        //	{
        //		int index = GetIdentifer().id(key); Float value = DOUBLE(value_);
        //		if (index != Identifer::null_id() && _selectedArray[index])
        //			features.Add(index, value);
        //	}
        //	else
        //	{
        //		//@TODO 暂时没有检查格式正确性 要求的是 所有非:的数据 都在后面
        //		switch (_columnTypes[j])
        //		{
        //		case ColumnType::Name:
        //			instance.names.push_back(item);
        //			break;
        //		case ColumnType::Label:
        //			instance.label = DOUBLE(item);
        //			break;
        //		case ColumnType::Weight:
        //			instance.weight = DOUBLE(item);
        //			break;
        //		case ColumnType::Attribute:
        //			instance.attributes.push_back(item);
        //			break;
        //		default:
        //			break;
        //		}
        //	}
        //}
      }
    }

    //1 张无忌:3.5 小甜甜:2.0  暂时没有测试 不推荐使用 尽量还是外部使用Identifer 脚本处理这些
    void CreateInstancesFromTextFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromTextFormat";
      int times = TextFormatParsedTime();
      if (times == 0)
        ParseTextForTrain(lines, start);
      else
        ParseTextForTest(lines, start);
    }

    //@TODO better  如果类别是 -2, -1, 2, 3 这样会有问题  默认只是  0, 1, 2, 3
    void CheckNumLabelsAndFix()
    {
      set<int> labels;
      bool isRegression = false;
      int maxLabel = 1;
      for (auto& instance : _instances)
      {
        int label = int(instance->label);
        if (label != instance->label)
        {
          isRegression = true;
          break;
        }
        if (label > maxLabel)
        {
          maxLabel = label;
        }
        labels.insert(label);
      }
      if (isRegression)
      {
        _instances.schema.numClasses = -1;
      }
      else
      {
        _instances.schema.numClasses = labels.size();
        if (maxLabel + 1 > labels.size())
        {
          LOG(WARNING) << "check if you have missing label data, as maxLabel is " << maxLabel << " but labels.size() is " << labels.size() << " which should be maxLabel + 1";
          _instances.schema.numClasses = maxLabel + 1;
        }
        if (labels.size() == 2 && labels.count(-1))
        {
          for (auto& instance : _instances)
          {
            if (instance->label == -1)
            {
              instance->label = 0;
            }
          }
        }
      }
    }



    void Clear()
    {
      _firstColums.clear();
      _headerColums.clear();
      _columnNum = 0;
    }

    //注意这个设计 一个global的identifer 同时意味生命期 最多两次parse text格式文本
    //parse 一次 比如 train, cross fold, 两次train test 不会更多次
    static Identifer& GetIdentifer()
    {
      static Identifer _identifer;
      return _identifer;
    }

    static int TextFormatParsedTime()
    {
      static int time = 0;
      return time++;
    }

    //------for malloc rank data format   #    label, qid:0  libsvm-features
    void TryAdaptForMallocRankFormat(svec& lines)
    {
      //malloc rank需要自动解析 不带任何其他输入信息
      if (_args.namesIdx.empty() && _args.attrsIdx.empty()
        && _args.labelIdx == -1 &&
        _args.groupsIdx.empty() && gezi::contains(lines[0], "qid:"))
      {
        _fileFormat = FileFormat::MallocRank;
        _namesIdx.push_back(1);
        //_attributesIdx.push_back(1);
        _groupsIdx.push_back(1);
        _labelIdx = 0;
      }
    }
  protected:
    Instances&& Parse_(string dataFile)
    {
      Timer timer;
      //如果有cache数据 直接从cache数据load
      string cacheFile = GetOutputFileNameWithSuffix(dataFile, "cache", true);
      if (bfs::exists(cacheFile))
      {
        if (bfs::last_write_time(cacheFile) > bfs::last_write_time(dataFile))
        {
          VLOG(0) << "Cache file exist, reading directly from " << cacheFile << " instead reading from " << dataFile;
          serialize_util::load(_instances, cacheFile);
          return move(_instances);
        }
      }
      vector<string> lines = read_lines_fast(dataFile, "//");
      TryAdaptForMallocRankFormat(lines); //malloc格式数据的补丁

      if (lines.empty())
      {
        LOG(FATAL) << "Fail to load data file! " << dataFile << " is empty!";
      }

      _instanceNum = lines.size();

      if (_args.hasHeader)
      {
        _instanceNum--;
        if (!_instanceNum)
        {
          LOG(FATAL) << "Only header no data! " << dataFile;
        }
        _hasHeader = true;
      }

      timer.restart();
      ParseFirstLine(lines);
      PVAL_(timer.elapsed_ms(), "ParseFirstLine");

      timer.restart();
      if (_numFeatures == 0)
      { //如果是没有指明Length的文本格式包括libsvm格式 都是没有指明 这时候仍然不确定特征数目
        //尝试设置一个较大的默认值 后续会再修正的 注意过滤的时候不要按照匹配了 按照index设定
        //libsvm格式的也按照melt标准0开始匹配
        _numFeatures = std::numeric_limits<int>::max();
      }
      {
        _selectedArray = GetSelectedArray();
        PVAL_(timer.elapsed_ms(), "GetSelectedArray");
      }

      _instanceNum = lines.size() - _hasHeader; //必须有 因为_hasHeader可能通过解析变为true
      if (!_instanceNum)
      {
        LOG(FATAL) << "Only header no data! " << dataFile;
      }
      _instances.resize(_instanceNum, nullptr);

      timer.restart();

      switch (_fileFormat)
      {
      case FileFormat::Dense:
        CreateInstancesFromDenseFormat(lines, _hasHeader);
        break;
      case FileFormat::Sparse:
        CreateInstancesFromSparseFormat(lines, _hasHeader);
        break;
      case FileFormat::SparseNoLength:
        CreateInstancesFromSparseNoLengthFormat(lines, _hasHeader);
        break;
      case  FileFormat::LibSVM:
        CreateInstancesFromLibSVMFormat(lines, _hasHeader);
        break;
      case  FileFormat::VW:
        CreateInstancesFromVWFormat(lines, _hasHeader);
        break;
      case FileFormat::Text:
        CreateInstancesFromTextFormat(lines, _hasHeader);
        break;
      case  FileFormat::MallocRank:
        CreateInstancesFromMallocRankFormat(lines, _hasHeader);
        break;
      default:
        LOG(WARNING) << "well not supported file format ?";
        break;
      }

      PVAL_(timer.elapsed_ms(), "CreateInstances");

      if (_args.cacheInstance)
      {
        serialize_util::save(_instances, cacheFile);
      }
      else
      {
        if (bfs::exists(cacheFile))
        {
          bfs::remove(cacheFile);
        }
        if (timer.elapsed() > 60)
        {
          VLOG(0) << "Loading big data file slow, you may try to use --cacheInst=1 to generate cache file so next time loading will be faster";
        }
      }

      return move(_instances);
    }

    //如果是libsvm格式 但是稠密格式数据 可能会占用较多内存 最后才会做Densify
    void FinallizeEachInstance()
    {
      VLOG(2) << "Before adjust for dense/sparse, dense count:" << _instances.DenseCount();
#pragma omp parallel for 
      for (uint64 i = 0; i < _instanceNum; i++)
      {
        //--完成名字设置
        _instances[i]->name = join(_instances[i]->names, _args.ncsep);
        if (startswith(_instances[i]->name, '_'))
        {
          _instances[i]->name = _instances[i]->name.substr(1);
        }
        Vector& features = _instances[i]->features;
        features.SetLength(_numFeatures);

        //--尝试稠密稀疏的转换
        if (features.IsDense())
        {
          if (features.keepSparse)
          {
            features.ToSparse();
          }
          else
          {
            features.Sparsify(_args.sparsifyThre);
          }
        }
        else
        {
          if (features.keepDense)
          {
            features.ToDense();
          }
          else
          {
            features.Densify(_args.sparsifyThre);
          }
        }

      }
      VLOG(2) << "After adjust for dense/sparse, dense count:" << _instances.DenseCount();
    }
    void Finallize()
    {
      FinallizeEachInstance();
      CheckNumLabelsAndFix();
      if (_instances.IsRankingInstances())
      {
        _instances.groups = _instances.GetGroups();
      }
    }

  private:
    Instances _instances;
    uint64 _instanceNum = 0;
    int _numFeatures = 0;
    bool _hasHeader = false;
    bool _hasWeight = false;
    FileFormat _fileFormat = FileFormat::Unknown;

    svec _firstColums;
    svec _headerColums;

    BitArray _selectedArray;
    vector<ColumnType> _columnTypes;
    vector<bool> _groupKeysMark;
    int _columnNum = 0;
    int _labelIdx = -1;

    //----params
    Arguments _args;
    string _sep;
    ivec _namesIdx;
    ivec _attributesIdx;
    ivec _groupsIdx;
    string _format;
  };

}  //----end of namespace gezi

#endif  //----end of PREDICTION__INSTANCES__INSTANCE_PARSER_H_
