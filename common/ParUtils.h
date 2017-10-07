#ifndef PAR_UTILS_H
#define PAR_UTILS_H
#include "Utils.h"

#include <thread>

template<class T>
class ParUtils : public Utils<T>{
public:
	ParUtils(){ this->ths = 1; };
	ParUtils(unsigned int ths){ this->ths = ths; };
	~ParUtils(){};

	void mtfast_readcsv(T *&arr, std::string file, arr2D dim);
	void mtfast_readcsv(T *&arr, std::string file, unsigned int d, unsigned int n);
	void setp(unsigned int ths){ this->ths = ths; }
private:
	unsigned int ths;
};

/***********************/
/*	ParUtils Class	   */
/***********************/

template<class T>
void ths_proc_mem(T *&arr, char *ptr_start, char *ptr_end, uint64_t first, std::string delim){
	uint64_t size = ptr_end - ptr_start + 1;
	char *buf = new char[size];
	memcpy(buf, ptr_start, size);
	buf[size - 1] = '\0';

	char *pch = std::strtok(buf, (delim + "\n").c_str());
	std::stringstream iss;
	uint64_t elem = 0;
	while (pch != NULL){
		elem++; iss << pch; iss << " ";
		pch = std::strtok(NULL, (delim + "\n").c_str());
	}

	uint64_t i = 0;
	while (i < elem){ iss >> arr[first + i]; i++; }
	delete buf;
}

template<class T>
void ParUtils<T>::mtfast_readcsv(T *&arr, std::string file, arr2D dim){
	this->mtfast_readcsv(arr, file, dim.first, dim.second);
}

template<class T>
void ParUtils <T>::mtfast_readcsv(T *&arr, std::string file, unsigned int d, unsigned int n){
	uint64_t totalbytes = this->fsize(file);
	FILE *fp = fopen(file.c_str(), "r");
	char *buffer = new char[totalbytes];
	//std::cout << "totalbytes: " << totalbytes << std::endl;
	totalbytes = fread(buffer, 1, totalbytes, fp);
	buffer[totalbytes] = '\0';
	fclose(fp);

	std::vector<std::pair<char*, uint64_t>> ths_data;
	ths_data.push_back(std::pair<char*, uint64_t>(buffer, 0));
	unsigned int chunk = (n - 1) / ths + 1;
	unsigned int lines = 1;
	char *pch = strchr(buffer, '\n');

	while (pch != NULL){
		if (lines % chunk == 0){ ths_data.push_back(std::pair<char*, uint64_t>(pch, lines*d)); }
		pch = strchr(pch + 1, '\n');
		lines++;
	}

	if (lines != n){ std::cout << "Line Number Error (" << lines << "," << n << ")" << std::endl;  return; }

	std::vector<std::thread> threads;
	for (int i = 0; i < ths_data.size(); i++){
		char *start = ths_data[i].first;
		char *end = i == ths_data.size() - 1 ? &buffer[totalbytes] : ths_data[i + 1].first;
		//ths_proc_mem<T>(arr, start, end, ths_data[i].second);
		threads.push_back(std::thread(ths_proc_mem<T>, arr, start, end, ths_data[i].second, this->delim));
	}
	for (auto& th : threads) th.join();

	delete buffer;
}


#endif