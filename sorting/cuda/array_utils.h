#include <iostream>

template<typename T> 
class Generator {
	public:
		void random(std::vector<T> &a, uint32_t size);
		void random(std::vector<T> &a, uint32_t size, long long maxNumber);
		void decreasing(std::vector<T> &a, uint32_t size);
		void increasing(std::vector<T> &a, uint32_t size);
};
