/**
*  ==============================================================================
*
*          \file   Prediction/Instances/InstanceParser.h
*
*        \author   chenghuige
*
*          \date   2014-03-27 17:47:57.438203
*
*  \Description:   �����������ļ� ������dense ���� sparse��ʽ �����Instances
*                  @TODOĿǰֻ����Instances�������鶼�������ڴ� ������ StreamingInstances @TODO
* �������ݸ�ʽ dense (default)
*  //with header
*	#id label age weight
*  _123 0    32   18
*  //whithout header
*  _123 0  32 18 //id label feature1 feature2
* �������ݸ�ʽ sparse (default)
*   0 5 0:0.4 3:0.1
*   1 5 1:0.3
* �ڲ����ݴ����TLCһ�� ����������dense����sparse�����жϵ�ǰ0����Ŀ ���> 1/2 * FeatureNum ��dense����sparse
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

  //@TODO �޸����ֻ�������namespace �׳�ͻ
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

  //RunConvert ��ʾת��ʹ��
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

  //���������� ʹ��
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


  //����clear������ÿ�� create һ���µ�InstanceParserʹ�ü���
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

    //Ĭ��������dense���ݣ�����û��header,������ ���fake header ��outFile
    void AddFakeHeader(string dataFile, string outFile)
    {
      //@TODO streaming ?
      vector<string> lines = read_lines_fast(dataFile, "//");
      if (lines.empty())
      {
        LOG(FATAL) << "Fail to load data file! " << dataFile << " is empty!";
      }

      //�ж�schema��Ϣ
      ParseFirstLine(lines);


      ofstream ofs(outFile);

      //-----------------дfake header
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

    //@TODO ��ǰֵ����֧��һ�� ���� i, e, s, r�����Ƕ��ͬʱ����,һ�� ��õ�r ����ƥ�伴��
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

      //ǰ��ȷ�����Ƿ���header
      if (_hasHeader)
      {
        _headerColums.swap(_firstColums);
        _firstColums = split(lines[_hasHeader], _sep);
      }

      //�����_��ͷ��ô��һ����Name ���û������labelIdx��ô�ڶ�����������Ϊ��Label
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

      //Ĭ�����û���ƶ�labelIdx ��һ����Label, �����name,attr���ƶ� ��ôname,attr����ĵ�һ����Label
      if (_labelIdx < 0)
      {//��� maxNameAtrrIdx == -1 , then 0
        _labelIdx = maxNameAtrrIdx + 1;
      }

      Pval(_labelIdx);
      _columnTypes[_labelIdx] = ColumnType::Label;
    }

    //ʵ����Dense,Sparse��ʽ ���������������, SparseNoLength, Text������create instances������
    void InitNames()
    {
      if (_hasHeader)
      { //��ζ���ļ�����ȷָ�������е���������
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
      { //��Ҫͳ��������Ŀ
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
        //{ //ע���Ѿ������׸������� ��ȡ�� ������Ŀ��(_numFeatures) ����ע�� SparseNoLength, Text��ȡ����
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
            double value; //���뵥��һ�С��� ����crosses initialization
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
            double value; //���뵥��һ�С��� ����crosses initialization
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
        ////CHECK_EQ(l.size(), _columnNum) << "has bad line " << i; //�������л�������
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

    //����ŵ������Ӱ���Ҳ�����������stl_util.h���涨���split...
    ////@TODO ����汾�� int + double �������Ҫ�������͵� fast atou��һЩ �������������� �ݲ�ʹ��
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
    //		if (inPos < pos2) //������ҿ����Ż� �������ⲻ�󡣡�
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
    //@TODO Instance Next(string line) ֧��streaming
    //@TODO tlcò�ƿ�ܶ�thread.feature.txt 9w instance 
    //1000ms ������Ҫ2000ms����Ϊsplit c#���ٶȸ��죿 ����ע����char split find��Ҫ��string
    //����ʹ��omp���к� 200ms�͸㶨
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
        //		//@TODO ��ʱû�м���ʽ��ȷ�� Ҫ����� ���з�:������ ���ں���
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

    //���û����ǰ����������󳤶�SparseNoLength ��ô��֧������ѡ��,����Ӱ������ٶ� ������ת��Ϊ������sparse��ʽ
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
              //#pragma omp critical //���Ĵ��� �Ƿ�ֵ�ò���@TODO
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
              //#pragma omp critical //���Ĵ��� �Ƿ�ֵ�ò���@TODO
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
        //						//@TODO ��ʱû�м���ʽ��ȷ�� Ҫ����� ���з�:������ ���ں��� ��������ϻ�Խ��
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
            //return FileFormat::SparseNoLength; //ȥ��Ĭ�Ͻ��� sparse no length��Ϊ������,Ϊ�˸��ü���libsvm��ʽ
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
              _numFeatures = INT(_firstColums[i]); //sparse��ʽ������������Ŀ�� @IMPORTANT
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

    //��ȡ����Ϣ�����֣���dense����sparse��ʾ
    void ParseFirstLine(const svec& lines)
    {
      //Timer timer;
      string line = lines[0];
      _firstColums = split(line, _sep);

      if (_firstColums.size() < 2)
      {
        char sep = GuessSeparator(lines[0], "\t ,");
        _firstColums = split(line, sep);
        //_sep = string(sep); //ò������char to string ��core����
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

      //������label��ӡÿ��label��instance��Ŀ��Ϣ
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

      //�����Ranking��ӡ�ж�����
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

    //����������������
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


    //@TODO �������Ż��������ȴ��� ������:������ Ȼ���д�����е�
    //@FIXM Ŀǰֻ��label���ݵ� Ҳ���ǿ������Ļ�core
    void CreateInstancesFromLibSVMFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromLibSVMFormat";
      uint64 end = start + _instanceNum;
      int maxIndex = _args.libsvmStartIndex;
      //char sep = GuessSeparator(lines[0], "\t "); //�Ѿ���ParseFirstLine��ʱ��ȷ����
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
        //				svec l = split(line, "\t "); //libsvm ���ÿո����tab���п���
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
        //					features.Add(index - 1, value); //libsvm ��1��ʼ melt/tlc�ڲ�0��ʼ����
        //				}
      }
      _numFeatures = _args.libsvmStartIndex == 1 ? maxIndex : maxIndex + 1;
      Rabit::Allreduce<op::Max>(_numFeatures);
      _instances.schema.featureNames.SetNumFeatures(_numFeatures);
    }

    //���malloc rank��ʽ���ݵ� �ָ��������
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
        if (inPos < pos2) //������ҿ����Ż� �������ⲻ�󡣡���ʱû�ҵ��õķ��� ��Ϊfind_char�ٶȸ����ܶ�
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

    //����libsvm ���ܳ��ܻ���ϡ�� ������ ��һ�� qid:0 ���Ƶ���featureǰ��
    void CreateInstancesFromMallocRankFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromMallocRankFormat";
      uint64 end = start + _instanceNum;
      int maxIndex = 1; //malloc ���� �±�1��ʼ
      //char sep = GuessSeparator(lines[0], "\t "); //�Ѿ���ParseFirstLine��ʱ��ȷ����
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

    //ѵ���ı����� ��ʱ����ֻ֧�ֵ�һchar�ķָ��� û�б�Ҫ֧��string�ָ�
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
            { //������Ҫ��֤û���ظ���key
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
            { //������Ҫ��֤û���ظ���key
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
        //							//������Ҫ��֤û���ظ���key
        //							index = GetIdentifer().add(key);
        //						}
        //						if (_selectedArray[index])
        //							features.Add(index, value);
        //					}
        //					else
        //					{
        //						//@TODO ��ʱû�м���ʽ��ȷ�� Ҫ����� ���з�:������ ���ں���
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

    //�����ı�����
    void ParseTextForTest(const svec& lines, uint64 start)
    {
      VLOG(0) << "ParseTextForTest";
      _numFeatures = GetIdentifer().size();
      if (_numFeatures == 0)
      { //����RunTestģʽ ��Ҫ���شʱ�
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
        //		//@TODO ��ʱû�м���ʽ��ȷ�� Ҫ����� ���з�:������ ���ں���
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

    //1 ���޼�:3.5 С����:2.0  ��ʱû�в��� ���Ƽ�ʹ�� ���������ⲿʹ��Identifer �ű�������Щ
    void CreateInstancesFromTextFormat(const svec& lines, uint64 start)
    {
      VLOG(0) << "CreateInstancesFromTextFormat";
      int times = TextFormatParsedTime();
      if (times == 0)
        ParseTextForTrain(lines, start);
      else
        ParseTextForTest(lines, start);
    }

    //@TODO better  �������� -2, -1, 2, 3 ������������  Ĭ��ֻ��  0, 1, 2, 3
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

    //ע�������� һ��global��identifer ͬʱ��ζ������ �������parse text��ʽ�ı�
    //parse һ�� ���� train, cross fold, ����train test ��������
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
      //malloc rank��Ҫ�Զ����� �����κ�����������Ϣ
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
      //�����cache���� ֱ�Ӵ�cache����load
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
      TryAdaptForMallocRankFormat(lines); //malloc��ʽ���ݵĲ���

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
      { //�����û��ָ��Length���ı���ʽ����libsvm��ʽ ����û��ָ�� ��ʱ����Ȼ��ȷ��������Ŀ
        //��������һ���ϴ��Ĭ��ֵ �������������� ע����˵�ʱ��Ҫ����ƥ���� ����index�趨
        //libsvm��ʽ��Ҳ����melt��׼0��ʼƥ��
        _numFeatures = std::numeric_limits<int>::max();
      }
      {
        _selectedArray = GetSelectedArray();
        PVAL_(timer.elapsed_ms(), "GetSelectedArray");
      }

      _instanceNum = lines.size() - _hasHeader; //������ ��Ϊ_hasHeader����ͨ��������Ϊtrue
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

    //�����libsvm��ʽ ���ǳ��ܸ�ʽ���� ���ܻ�ռ�ý϶��ڴ� ���Ż���Densify
    void FinallizeEachInstance()
    {
      VLOG(2) << "Before adjust for dense/sparse, dense count:" << _instances.DenseCount();
#pragma omp parallel for 
      for (uint64 i = 0; i < _instanceNum; i++)
      {
        //--�����������
        _instances[i]->name = join(_instances[i]->names, _args.ncsep);
        if (startswith(_instances[i]->name, '_'))
        {
          _instances[i]->name = _instances[i]->name.substr(1);
        }
        Vector& features = _instances[i]->features;
        features.SetLength(_numFeatures);

        //--���Գ���ϡ���ת��
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
