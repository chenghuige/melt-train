/**
 *  ==============================================================================
 *
 *          \file   Prediction/Instances/instances_util.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-04-02 20:34:03.488226
 *
 *  \Description: @FIXME Ӧ���� InstancesUtil.h
 *  ==============================================================================
 */

#ifndef PREDICTION__INSTANCES_INSTANCES_UTIL_H_
#define PREDICTION__INSTANCES_INSTANCES_UTIL_H_

#include "InstanceParser.h"
#include "random_util.h"

DECLARE_int32(libsvmNL);
DECLARE_int32(libsvmSI);
namespace gezi {

	//@TODO ��Ҫ���ӵ� boot strap��ʽ�õ�һ���µ�Insatnces, Shrink�ķ�ʽ�õ�һ���µ�
	class InstancesUtil
	{
	public:
		static void SplitInstancesByLabel(const Instances& instances, Instances& posInstances, Instances& negInstances)
		{
			for (InstancePtr instance : instances)
			{
				if (instance->IsPositive())
					posInstances.push_back(instance);
				else
					negInstances.push_back(instance);
			}
		}

		static vector<Instances> RandomSplit(const Instances& instances, double ratio, unsigned randSeed = 0)
		{
			RandomEngine rng = random_engine(randSeed);
			instances.Randomize(rng);
			size_t numTotalInsatnces = instances.size();
			size_t numFirstPartInstances = (size_t)(numTotalInsatnces * (1 - ratio));
			vector<Instances> parts(2);
			parts[0].CopySchema(instances.schema);
			parts[1].CopySchema(instances.schema);

			for (size_t i = 0; i < numFirstPartInstances; i++)
			{
				parts[0].push_back(instances[i]);
			}

			for (size_t i = numFirstPartInstances; i < numTotalInsatnces; i++)
			{
				parts[1].push_back(instances[i]);
			}
			return parts;
		}

		static	Instances GenPartionInstances(const Instances& instances, const Random& rand, double fraction)
		{
			if (fraction >= 1.0)
			{
				return instances;
			}
			Instances partitionInstaces;
			partitionInstaces.CopySchema(instances.schema);
			for (const auto& instance : instances)
			{
				if (rand.NextDouble() < fraction)
				{
					partitionInstaces.push_back(instance);
				}
			}
			return partitionInstaces;
		}

		static Instances GenBootstrapInstances(const Instances& instances, const Random& rand, double fraction)
		{
			size_t nowCount = instances.size();
			size_t destCount = nowCount * fraction;
			Instances destInstaces;
			destInstaces.CopySchema(instances.schema);
			for (size_t i = 0; i < destCount; i++)
			{
				int idx = rand.Next((int)nowCount);
				destInstaces.push_back(instances[idx]);
			}
			return destInstaces;
		}

		static InstancePtr SampleOneInstance(const Instances& instances, const Random& rand)
		{
			size_t idx = rand.Next(instances.size());
			return instances[idx];
		}
	protected:
	private:
	};

	//@TODO���治���޸� ����ʵ�����ʹ��InstacesUtil::CreateInstance InstancesUtil::Write����һЩ
	inline Instances create_instances(string infile, bool printInfo = false)
	{
		InstanceParser parser;
		return parser.Parse(infile, printInfo);
	}

	//ע���޸���intance �ر������clear�� sparse תdense ��δ��֤
	inline void write_dense(Instance& instance, HeaderSchema& schema, ofstream& ofs)
	{
		instance.features.MakeDense(); //�޸��� ����instance �����ת�� []������������Ҳ������ForEachAll
		size_t featureIdx = 0, nameIdx = 0, attributeIdx = 0;
		switch (schema.cloumnTypes[0])
		{
		case ColumnType::Feature:
			ofs << instance.features[featureIdx++];
			break;
		case ColumnType::Name:
			ofs << instance.names[nameIdx++];
			break;
		case ColumnType::Label:
			ofs << instance.label;
			break;
		case ColumnType::Weight:
			ofs << instance.weight;
			break;
		case ColumnType::Attribute:
			ofs << instance.attributes[attributeIdx++];
			break;
		default:
			break;
		}
		for (size_t i = 1; i < schema.cloumnTypes.size(); i++)
		{
			switch (schema.cloumnTypes[i])
			{
			case ColumnType::Feature:
				ofs << "\t" << instance.features[featureIdx++];
				break;
			case ColumnType::Name:
				ofs << "\t" << instance.names[nameIdx++];
				break;
			case ColumnType::Label:
				ofs << "\t" << instance.label;
				break;
			case ColumnType::Weight:
				ofs << "\t" << instance.weight;
				break;
			case ColumnType::Attribute:
				ofs << "\t" << instance.attributes[attributeIdx++];
				break;
			default:
				break;
			}
		}
		//ϡ��תdesne�ܿ���Ҫ�����������
		for (; featureIdx < instance.features.Length(); featureIdx++)
		{
			ofs << "\t" << instance.features[featureIdx];
		}
		ofs << endl;
		instance.features.Clear(); //���ⶼתdense�����ڴ�����
	}

	//��heder dense �Ѿ���֤ok
	inline void write_dense(Instances& instances, string outfile)
	{
		ofstream ofs(outfile);
		if (instances.HasHeader())
		{
			ofs << instances.HeaderStr() << endl;
		}
		for (InstancePtr instance : instances)
		{
			write_dense(*instance, instances.schema, ofs);
		}
	}

	//sparse �Լ�ת��ok  ����Ҫע�� �����dense תsparse Ҫȷ�� desne��feature �������������Ժ����
	inline void write_sparse(Instance& instance, HeaderSchema& schema, ofstream& ofs)
	{
		size_t featureIdx = 0, nameIdx = 0, attributeIdx = 0;
		switch (schema.cloumnTypes[0])
		{
		case ColumnType::Feature:
			ofs << instance.features.indices[featureIdx++] << ":" << instance.features.values[featureIdx++];
			break;
		case ColumnType::Name:
			ofs << instance.names[nameIdx++];
			break;
		case ColumnType::Label:
			ofs << instance.label;
			break;
		case ColumnType::Weight:
			ofs << instance.weight;
			break;
		case ColumnType::Attribute:
			ofs << instance.attributes[attributeIdx++];
			break;
		default:
			break;
		}

		for (size_t i = 1; i < schema.cloumnTypes.size(); i++)
		{
			if (schema.cloumnTypes[i] == ColumnType::Feature)
			{
				break;
			}
			switch (schema.cloumnTypes[i])
			{
			case ColumnType::Name:
				ofs << "\t" << instance.names[nameIdx++];
				break;
			case ColumnType::Label:
				ofs << "\t" << instance.label;
				break;
			case ColumnType::Weight:
				ofs << "\t" << instance.weight;
				break;
			case ColumnType::Attribute:
				ofs << "\t" << instance.attributes[attributeIdx++];
				break;
			default:
				break;
			}
		}

		if (schema.fileFormat != FileFormat::Sparse)
		{//Sparse��ʽ�Ļ� �Ѿ�����Attribute��¼��������Ŀ, ��Ȼһ�㲻��Ҫsparse->sparse����debug
			ofs << "\t" << instance.NumFeatures();
		}

		instance.features.ForEachNonZero([&ofs](int index, Float value)
		{
			ofs << "\t" << index << ":" << value;
		});

		if (instance.features.NumNonZeros() == 0)
		{
			ofs << "\t0:0.0";
		}

		ofs << endl;
	}

	inline void write_sparse(Instances& instances, string outfile)
	{
		ofstream ofs(outfile);
		if (instances.HasHeader())
		{
			ofs << instances.HeaderStr() << endl;
		}
		for (InstancePtr instance : instances)
		{
			write_sparse(*instance, instances.schema, ofs);
		}
	}

	inline void write_text(Instances& instances, string outfile)
	{

	}

	//��ʱ��������δ��עlabel����� δ��ע����Ϊ-1 normal ����
	inline void write_libsvm(Instance& instance, HeaderSchema& schema, ofstream& ofs, bool noNegLabel = false, int startIndex = 1, bool isVW = false)
	{
		if (schema.numClasses == 2)
		{ //Ϊ��sofia���� ��0תΪ-1 ����libsvm sofia������ֱ�Ӵ������ָ�ʽ
			if (noNegLabel)
			{
				if (instance.label == -std::numeric_limits<double>::infinity())
				{
					ofs << 0;
				}
				else
				{
					ofs << instance.label;
				}
			}
			else
			{
				if (instance.label == -std::numeric_limits<double>::infinity() ||
					instance.label == 0)
				{
					ofs << -1;
				}
				else
				{
					ofs << instance.label;
				}
			}
		}
		else
		{
			ofs << instance.label;
		}

		if (isVW)
		{
			ofs << " |n";
		}

		instance.features.ForEachNonZero([&](int index, Float value)
		{
			ofs << " " << index + startIndex << ":" << value;
		});

		if (instance.features.NumNonZeros() == 0)
		{
			ofs << " 1:0.0";
		}
		ofs << endl;
	}

	inline void write_libsvm(Instances& instances, string outfile, bool noNegLabel = false, int startIndex = 1)
	{
		ofstream ofs(outfile);
		for (InstancePtr instance : instances)
		{
			write_libsvm(*instance, instances.schema, ofs, noNegLabel, startIndex);
		}
	}

	inline void write_vw(Instances& instances, string outfile)
	{
		ofstream ofs(outfile);
		for (InstancePtr instance : instances)
		{
			write_libsvm(*instance, instances.schema, ofs, false, true);
		}
	}

	inline void write_arff(Instances& instances, string outfile, string relation = "table")
	{
		ofstream ofs(outfile);
		//----------arff header
		ofs << "@relation " << relation << "\n" << endl;
		for (string name : instances.schema.featureNames)
		{
			ofs << "@attribute " << name << " numeric" << endl;
		}

		ofs << "@attribute class {" << "negative,positive" << "}\n" << endl;

		ofs << "@data\n" << endl;

		//----------write body
		svec types = { "negative", "positive" };
		for (InstancePtr instance : instances)
		{
			ofs << "{";
			instance->features.ForEachNonZero([&ofs](int index, Float value)
			{
				ofs << index << " " << value << ",";
			});
			int index = instance->features.Length();
			ofs << index << " " << types[instance->IsPositive()];
			ofs << "}" << endl;
		}
	}

	inline void write(Instances& instances, string outfile, FileFormat format)
	{
		if (format == FileFormat::Unknown)
		{
			format = instances.schema.fileFormat;
		}
		VLOG(0) << "Writing to: " << outfile << " in format: " << kFormatNames[format];

		switch (format)
		{
		case FileFormat::Dense:
			write_dense(instances, outfile);
			break;
		case  FileFormat::Sparse:
		case FileFormat::SparseNoLength:
			write_sparse(instances, outfile);
			break;
		case FileFormat::Text:
			break;
		case  FileFormat::LibSVM:
			write_libsvm(instances, outfile, FLAGS_libsvmNL == 0, FLAGS_libsvmSI);
			break;
		case FileFormat::Arff:
			write_arff(instances, outfile);
			break;
		case  FileFormat::VW:
			write_vw(instances, outfile);
			break;
		default:
			break;
		}
	}

	inline void write(Instances& instances, string outfile)
	{
		write(instances, outfile, instances.schema.fileFormat);
	}
}  //----end of namespace gezi

#endif  //----end of PREDICTION__INSTANCES_INSTANCES_UTIL_H_
