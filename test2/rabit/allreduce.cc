/**
 *  ==============================================================================
 *
 *          \file   allreduce.cc
 *
 *        \author   chenghuige
 *
 *          \date   2015-06-12 11:37:58.447771
 *
 *  \Description:
 *
 *  ==============================================================================
 */

#define private public
#define protected public
#include <rabit.h>
using namespace rabit;
#include "common_util.h"
#include "rabit_util.h"
const int N = 3;

using namespace std;
using namespace gezi;

DEFINE_int32(vl, 0, "vlog level");
DEFINE_int32(level, 0, "min log level");
DEFINE_string(type, "simple", "");
DEFINE_bool(perf, false, "");
DEFINE_int32(num, 1, "");
DEFINE_string(i, "", "input file");
DEFINE_string(o, "", "output file");


struct Node : public RabitObject
{
	int x;
	int y;
};

inline void UpdateNode(Node& dest, const Node& src)
{
	dest.x = std::max(dest.x, src.x);
	dest.y = std::min(dest.y, src.y);
}

inline void UpdateVec(ivec& dest, const ivec& src)
{
	if (dest.empty() && !src.empty())
	{
		dest = src;
	}
}

//allreduce.cc:(.text + 0x360) : undefined reference to `rabit: : engine::mpi::DataType rabit::engine::mpi::GetType<Node>()'
//struct NodeUpdate {
//	const static engine::mpi::OpType kType = engine::mpi::kMax;
//	inline static void Reduce(Node &dest, const Node &src) {
//		dest.x += src.x;
//		dest.y *= src.y;
//	}
//};
void run()
{
	int a[N];

	for (int i = 0; i < N; ++i) {
		a[i] = rabit::GetRank() + i;
	}
	printf("@node[%d] before-allreduce: a={%d, %d, %d}\n",
		rabit::GetRank(), a[0], a[1], a[2]);
	// allreduce take max of each elements in all processes
	Allreduce<op::Max>(&a[0], N);
	printf("@node[%d] after-allreduce-max: a={%d, %d, %d}\n",
		rabit::GetRank(), a[0], a[1], a[2]);
	// second allreduce that sums everything up
	Allreduce<op::Sum>(&a[0], N);
	printf("@node[%d] after-allreduce-sum: a={%d, %d, %d}\n",
		rabit::GetRank(), a[0], a[1], a[2]);
	{
		vector<Node> a(N);
		for (int i = 0; i < N; ++i) {
			a[i].x = rabit::GetRank() + 2 * i;
			a[i].y = rabit::GetRank() + i;
		}

		printf("<struct> @node[%d] before-allreduce: a.x={%d, %d, %d}\n",
			rabit::GetRank(), a[0].x, a[1].x, a[2].x);
		printf("@node[%d] before-allreduce: a.y={%d, %d, %d}\n",
			rabit::GetRank(), a[0].y, a[1].y, a[2].y);
		// allreduce take max of each elements in all processes
		//rabit::Reducer<Node, UpdateNode> reducer;
		//reducer.Allreduce(&a[0], N);
		ufo::Allreduce<Node, UpdateNode>(a);
		//ufo::Allreduce(a, UpdateNode); //not ok...
		printf("<struct> @node[%d] after-allreduce-max: a.x={%d, %d, %d}\n",
			rabit::GetRank(), a[0].x, a[1].x, a[2].x);
		printf("<struct> @node[%d] after-allreduce-max: a.y={%d, %d, %d}\n",
			rabit::GetRank(), a[0].y, a[1].y, a[2].y);
	}

	//----------尝试lambda失败
	//{
	//	vector<Node> a(N);
	//	for (int i = 0; i < N; ++i) {
	//		a[i].x = rabit::GetRank() + 2 * i;
	//		a[i].y = rabit::GetRank() + i;
	//	}

	//	printf("<struct> @node[%d] before-allreduce: a.x={%d, %d, %d}\n",
	//		rabit::GetRank(), a[0].x, a[1].x, a[2].x);
	//	printf("@node[%d] before-allreduce: a.y={%d, %d, %d}\n",
	//		rabit::GetRank(), a[0].y, a[1].y, a[2].y);
	//	// allreduce take max of each elements in all processes
	//	//rabit::Reducer<Node, UpdateNode> reducer;
	//	//reducer.Allreduce(&a[0], N);
	//	auto constexpr func = [](Node& dest, const Node& src)
	//	{
	//		dest.x = std::max(dest.x, src.x);
	//		dest.y = std::min(dest.y, src.y);
	//	};
	//	ufo::Allreduce<Node, func>(a);
	//	//ufo::Allreduce(a, UpdateNode); //not ok...
	//	printf("<struct> @node[%d] after-allreduce-max: a.x={%d, %d, %d}\n",
	//		rabit::GetRank(), a[0].x, a[1].x, a[2].x);
	//	printf("<struct> @node[%d] after-allreduce-max: a.y={%d, %d, %d}\n",
	//		rabit::GetRank(), a[0].y, a[1].y, a[2].y);
	//}

	{
		vector<Node> a(N);
		for (int i = 0; i < N; ++i) {
			if (Rabit::Choose(i))
			{
				a[i].x = rabit::GetRank() + 2 * i;
				a[i].y = rabit::GetRank() + i;
			}
		}

		printf("<Rabit> @node[%d] before-allreduce: a.x={%d, %d, %d}\n",
			rabit::GetRank(), a[0].x, a[1].x, a[2].x);
		printf("@node[%d] before-allreduce: a.y={%d, %d, %d}\n",
			rabit::GetRank(), a[0].y, a[1].y, a[2].y);

		Rabit::Allreduce(a);

		printf("<Rabit> @node[%d] after-allreduce-max: a.x={%d, %d, %d}\n",
			rabit::GetRank(), a[0].x, a[1].x, a[2].x);
		printf("<Rabit> @node[%d] after-allreduce-max: a.y={%d, %d, %d}\n",
			rabit::GetRank(), a[0].y, a[1].y, a[2].y);
	}

	//{
	//	vector<Node> a(N);
	//	for (int i = 0; i < N; ++i) {
	//		a[i].x = rabit::GetRank() + 2 * i;
	//		a[i].y = rabit::GetRank() + i;
	//	}

	//	printf("@node[%d] before-allreduce: a.x={%d, %d, %d}\n",
	//		rabit::GetRank(), a[0].x, a[1].x, a[2].x);
	//	printf("@node[%d] before-allreduce: a.y={%d, %d, %d}\n",
	//		rabit::GetRank(), a[0].y, a[1].y, a[2].y);
	//	// allreduce take max of each elements in all processes
	//	rabit::Allreduce<NodeUpdate>(&a[0], N);
	//	printf("@node[%d] after-allreduce-max: a.x={%d, %d, %d}\n",
	//		rabit::GetRank(), a[0].x, a[1].x, a[2].x);
	//	printf("@node[%d] after-allreduce-max: a.x={%d, %d, %d}\n",
	//		rabit::GetRank(), a[0].y, a[1].y, a[2].y);
	//}

	//allreduce.cc:(.text + 0x571) : undefined reference to `rabit: : engine::mpi::DataType rabit::engine::mpi::GetType<std::string>()'
	//{
	//	vector<string> a(N); 
	//	for (int i = 0; i < N; ++i) {
	//		a[i] = STR(rabit::GetRank() + i);
	//	}
	//	Pvec(a);
	//	rabit::Allreduce<op::Max>(&a[0], N);
	//	Pvec(a);
	//}

	{
		vector<ivec> values(rabit::GetWorldSize());
		if (rabit::GetRank() == 0)
		{
			values[0] = { 1, 2, 3 };
		}
		else
		{
			values[rabit::GetRank()] = { rabit::GetRank(), 4, 5, 6};
		}

		for (size_t i = 0; i < values.size(); i++)
		{
			Pvector(values[i]);
		}

		for (size_t i = 0; i < values.size(); i++)
		{
			rabit::Broadcast(&values[i], i);
		}
		VLOG(0) << "after allgather";

		for (size_t i = 0; i < values.size(); i++)
		{
			Pvector(values[i]);
		}
		ivec vec;
		for (size_t i = 0; i < values.size(); i++)
		{
			gezi::merge(vec, values[i]);
		}
		Pvector(vec);
	}
	{
		//对应vector<ivec>这个结果不对  @TODO
		vector<ivec> values(rabit::GetWorldSize());
		if (rabit::GetRank() == 0)
		{
			values[0] = { 1, 2, 3 };
		}
		else
		{
			values[rabit::GetRank()] = { rabit::GetRank(), 4, 5, 6 };
		}

		for (size_t i = 0; i < values.size(); i++)
		{
			Pvector(values[i]);
		}

		//rabit::Allreduce(&values[i], i);
		ufo::Allreduce<ivec, UpdateVec>(values);

		VLOG(0) << "after allgather";

		for (size_t i = 0; i < values.size(); i++)
		{
			Pvector(values[i]);
		}
		ivec vec;
		for (size_t i = 0; i < values.size(); i++)
		{
			gezi::merge(vec, values[i]);
		}
		Pvector(vec);
	}
	{
		ivec values;
		if (rabit::GetRank() == 0)
		{
			values = { 1, 2, 3 };
		}
		else
		{
			values = { rabit::GetRank(), 4, 5, 6 };
		}
		Pvector(values);
		Rabit::Allgather(values);
		Pvector(values);
	}
	{ //rabit不是线程安全的
#pragma omp parallel for
		for (size_t i = 0; i < 16; i++)
		{
			ivec values;

				if (rabit::GetRank() == 0)
				{
					values = { 1, 2, 3 };
				}
				else
				{
					values = { rabit::GetRank(), 4, 5, 6 };
				}
#pragma omp critical
			{
				Pvector(values);
				Rabit::Allgather(values);
				Pvector(values);
			}
		}
	}
}
//I0616 07:58 : 27.488692 20217 allreduce.cc : 196] values[i]:1 2 3
//I0616 07 : 58 : 27.488688 20218 allreduce.cc : 196] values[i]:
//I0616 07 : 58 : 27.488765 20217 allreduce.cc : 196] values[i]:
//I0616 07 : 58 : 27.488767 20218 allreduce.cc : 196] values[i]:0 : 1 1 : 4 2 : 5 3 : 6
//I0616 07 : 58 : 27.488920 20218 allreduce.cc : 203] after allgather
//I0616 07:58 : 27.488921 20217 allreduce.cc : 203] after allgather
//I0616 07:58 : 27.488934 20218 allreduce.cc : 207] values[i]:1 2 3
//I0616 07 : 58 : 27.488936 20217 allreduce.cc : 207] values[i]:1 2 3
//I0616 07 : 58 : 27.488942 20218 allreduce.cc : 207] values[i]:0 : 1 1 : 4 2 : 5 3 : 6
//I0616 07 : 58 : 27.488946 20217 allreduce.cc : 207] values[i]:0 : 1 1 : 4 2 : 5 3 : 6
//I0616 07 : 58 : 27.488953 20218 allreduce.cc : 214] vec:0 : 1 1 : 2 2 : 3 3 : 1 4 : 4 5 : 5 6 : 6
//I0616 07 : 58 : 27.488957 20217 allreduce.cc : 214] vec:0 : 1 1 : 2 2 : 3 3 : 1 4 : 4 5 : 5 6 : 6
//I0616 07 : 58 : 27.488967 20218 allreduce.cc : 229] values[i]:
//I0616 07 : 58 : 27.488971 20217 allreduce.cc : 229] values[i]:1 2 3
//I0616 07 : 58 : 27.488977 20218 allreduce.cc : 229] values[i]:0 : 1 1 : 4 2 : 5 3 : 6
//I0616 07 : 58 : 27.488979 20217 allreduce.cc : 229] values[i]:
//I0616 07 : 58 : 27.489029 20217 allreduce.cc : 235] after allgather
//I0616 07:58 : 27.489030 20218 allreduce.cc : 235] after allgather
//I0616 07:58 : 27.489042 20217 allreduce.cc : 239] values[i]:7815984 0 3
//I0616 07 : 58 : 27.489042 20218 allreduce.cc : 239] values[i]:1 4 5
//I0616 07 : 58 : 27.489050 20217 allreduce.cc : 239] values[i]:0 : 7816016 1 : 0 2 : 3 3 : 6
//I0616 07 : 58 : 27.489053 20218 allreduce.cc : 239] values[i]:0 : 1668572463 1 : 1668246575 2 : 1769237601 3 : 25965
//I0616 07 : 58 : 27.489061 20217 allreduce.cc : 246] vec:0 : 7815984 1 : 0 2 : 3 3 : 7815984 4 : 0 5 : 3 6 : 6
//I0616 07 : 58 : 27.489064 20218 allreduce.cc : 246] vec:0 : 1 1 : 4 2 : 5 3 : 1668572463 4 : 1668246575 5 : 1769237601 6 : 25965
int main(int argc, char *argv[])
{
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
	google::SetVersionString(get_version());
	int s = google::ParseCommandLineFlags(&argc, &argv, false);
	if (FLAGS_log_dir.empty())
		FLAGS_logtostderr = true;
	FLAGS_minloglevel = FLAGS_level;
	//LogHelper::set_level(FLAGS_level);
	if (FLAGS_v == 0)
		FLAGS_v = FLAGS_vl;

	Rabit rabit_(argc, argv);
	run();

	return 0;
}

