/**
 *  ==============================================================================
 *
 *          \file   MappedObjectPool.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-09 21:48:22.326876
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef MAPPED_OBJECT_POOL_H_
#define MAPPED_OBJECT_POOL_H_
#include "common_def.h"
namespace gezi {
	//本质是一个内存管理 如果不考虑内存大小 理想全部开辟10个空间 比如对应map size 
	//使用pool只开辟比如4个空间 通过map 定位的时候寻找合适的pool位置
	template<typename T>
	class MappedObjectPool
	{
	private:
		ivec _inverseMap;
		ivec _lastAccessTime;
		ivec _map;
		vector<T> _pool;
		int _time = 0;

	public:
		MappedObjectPool() = default;
		MappedObjectPool(vector<T>& pool, int maxIndex)
		{
			Initialize(pool, maxIndex);
		}

		void Initialize(vector<T>& pool, int maxIndex)
		{
			//PVAL(maxIndex);
			_pool.swap(pool);
			_map.resize(maxIndex, -1);
			_inverseMap.resize(_pool.size(), -1);
			_lastAccessTime.resize(_pool.size(), 0);
		}

		bool SimpleGet(int index, T*& obj)
		{
			obj = &_pool[index];
			return true;
		}

		bool Get(int index, T& obj)
		{
			if (_map[index] >= 0)
			{
				int position = _map[index];
				_lastAccessTime[position] = ++_time;
				obj = _pool[position];
				return true;
			}
			//尝试访问当前最少访问的位置
			int stealPosition = gezi::min_index(_lastAccessTime);
			_lastAccessTime[stealPosition] = ++_time;
			if (_inverseMap[stealPosition] >= 0)
			{ //标记之前映射到stealPostion的index失效
				_map[_inverseMap[stealPosition]] = -1;
			}
			_map[index] = stealPosition;
			_inverseMap[stealPosition] = index;
			obj = _pool[stealPosition];
			return false;
		}

		bool Get(int index, T*& obj)
		{
			if (_map[index] >= 0)
			{
				int position = _map[index];
				_lastAccessTime[position] = ++_time;
				obj = &_pool[position];
				return true;
			}
			int stealPosition = gezi::min_index(_lastAccessTime);
			_lastAccessTime[stealPosition] = ++_time;
			if (_inverseMap[stealPosition] >= 0)
			{
				_map[_inverseMap[stealPosition]] = -1;
			}
			_map[index] = stealPosition;
			_inverseMap[stealPosition] = index;
			obj = &_pool[stealPosition];
			return false;
		}

		void Reset()
		{
			gezi::zeroset(_lastAccessTime);
			_time = 0;
			ufo::fill(_map, -1);
			ufo::fill(_inverseMap, -1);
		}

		//将fromIndex的映射位置取到toIndex
		void Steal(int fromIndex, int toIndex)
		{
			if (_map[fromIndex] >= 0)
			{
				int stealPosition = _map[toIndex] = _map[fromIndex];
				_lastAccessTime[stealPosition] = ++_time;
				_inverseMap[stealPosition] = toIndex;
				_map[fromIndex] = -1;
			}
		}
	};
}  //----end of namespace gezi

#endif  //----end of MAPPED_OBJECT_POOL_H_
