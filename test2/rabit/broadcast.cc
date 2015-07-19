#include <rabit.h>
#include <vector>
#include "common_util.h"
using namespace rabit;
using namespace std;
const int N = 3;
struct Node
{
	int x;
	int y;
};
int main(int argc, char *argv[]) {
  rabit::Init(argc, argv);
  std::string s;
  if (rabit::GetRank() == 0) s = "hello world";
  printf("@node[%d] before-broadcast: s=\"%s\"\n",
         rabit::GetRank(), s.c_str());
  // broadcast s from node 0 to all other nodes
  rabit::Broadcast(&s, 0);
  printf("@node[%d] after-broadcast: s=\"%s\"\n",
		rabit::GetRank(), s.c_str());

	{
		vector<Node> vec;
		if (rabit::GetRank() == 0)
		{
			vec.resize(2);
			vec[0].x = 3;
			vec[0].y = 4;
		}
		rabit::Broadcast(&vec, 0);
		Pval4(vec[0].x, vec[0].y, vec[1].x, rabit::GetRank());
	}
  rabit::Finalize();
  return 0;
}
