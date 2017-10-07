#ifndef TOOLS_H
#define TOOL_H
/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * Various tools. (Beta version)
 */


#include <algorithm>

template<class K, class V>
class Tools{
public:
	Tools(){};
	~Tools(){};

	void sort(K*&, V*&, unsigned int s, unsigned int e);//e should be size - 1
};


/***********************/
/*	  Tools Class	   */
/***********************/

template<class K, class V>
void Tools<K, V>::sort(K *&arr, V *&val, unsigned int s, unsigned int e){
	unsigned int size = e - s;
	unsigned int splitpoint = s + size / 2;

	if (size == 0) return;
	if (size == 1){
		val[s] > val[e] ? std::swap(val[s], val[e]), std::swap(arr[s], arr[e]) : 0;
		return;
	}
	else{
		Tools<K, V>::sort(arr, val, s, splitpoint);
		Tools<K, V>::sort(arr, val, splitpoint + 1, e);
		int left = s;
		int right = splitpoint + 1;

		V *tmpA = new V[size + 1];
		K *tmpB = new K[size + 1];
		for (int i = 0; i < size + 1; i++){
			if (left > splitpoint){
				tmpA[i] = val[right];
				tmpB[i] = arr[right];
				right++;
			}
			else if (right > e){
				tmpA[i] = val[left];
				tmpB[i] = arr[left];
				left++;
			}
			else if (val[left] > val[right]){
				tmpA[i] = val[right];
				tmpB[i] = arr[right];
				right++;
			}
			else {
				tmpA[i] = val[left];
				tmpB[i] = arr[left];
				left++;
			}
		}
		memcpy(&arr[s], tmpB, sizeof(K)*(size + 1));
		memcpy(&val[s], tmpA, sizeof(V)*(size + 1));
		delete tmpA;
		delete tmpB;
	}
}

#endif
