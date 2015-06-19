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
	//������һ���ڴ���� ����������ڴ��С ����ȫ������10���ռ� �����Ӧmap size 
	//ʹ��poolֻ���ٱ���4���ռ� ͨ��map ��λ��ʱ��Ѱ�Һ��ʵ�poolλ��
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
			//���Է��ʵ�ǰ���ٷ��ʵ�λ��
			int stealPosition = gezi::min_index(_lastAccessTime);
			_lastAccessTime[stealPosition] = ++_time;
			if (_inverseMap[stealPosition] >= 0)
			{ //���֮ǰӳ�䵽stealPostion��indexʧЧ
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

		//��fromIndex��ӳ��λ��ȡ��toIndex
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
