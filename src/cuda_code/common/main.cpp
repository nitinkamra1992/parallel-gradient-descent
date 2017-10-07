/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * Testing main function.
 */

#include "ArgParser.h"

#include "Time.h"
//#include "CudaHelper.h"
#include "Utils.h"
#include "IOTools.h"

/*Functor Example*/
template<typename T>
T myFunc(T x , T y){
	return x*y + y;
}

template<typename T,typename FuncType>
T doMath (T x, T y, FuncType func)
{
    return func( x, y );
}

void test (){
	float a  = 0.12 , b = 0.12;
	float result = doMath<float>(a,b,myFunc<float>);
	std::cout<<"result<>: " << result << std::endl;
}

struct Default{
	template<typename T>
	inline T operator()(T x){
		return x;
	}
};

struct Sigmoid{
	template<typename T>
	inline T operator()(T x){
		return 1/(1 + exp(-x));
	}
};

struct FSigmoid{
	template<typename T>
	inline T operator()(T x){
		return x/(1.0 + fabs(x));
	}
};

template<typename T,typename ACT>
struct Layer{
	ACT F;
	T x;

	//Layer(ACT F){
	Layer(ACT F){
		this->F = F;
		x = 1.2;
	}

	void print(){
		std::cout<<x<<std::endl;
	}

	T compute(){
		return F(1.2);
	}

};

void test_template_function(){
	Default d;
	Sigmoid s;
	FSigmoid fs;

	std::cout<<d('x')<< std::endl;
	std::cout<<s(1.2)<< std::endl;
	std::cout<<fs(1.2)<< std::endl;

	Layer<float,Default> ld(d);
	Layer<float,Sigmoid> ls(s);
	Layer<float,FSigmoid> lfs(fs);

	float x = 1.2;
	std::cout<<"----\n";
	std::cout<<ld.compute()<< std::endl;
	std::cout<<ls.compute()<< std::endl;
	std::cout<<lfs.compute()<< std::endl;
}


void testReadWrite(ArgParser ap){
	int c = ap.getInt(CARG);
	int d = ap.getInt(DARG);

	if(ap.exists(MDARG)){
		static const char s[] = "Test 2";
		IOTools<float> iot;
		if(!ap.getString(MDARG).compare("w")){
			if(!ap.exists(DARG) ){
				vz::error("provide dimensionality");
			}

			if(!ap.exists(CARG)){
				vz::error("provide cardinality");
			}
			iot.randDataToFile("testData.csv",d,c,1024);

		}else if(!ap.getString(MDARG).compare("r")){
			float *data= NULL;
			iot.freadFile(data,"testData.csv",false);
		}
	}
}




int main(int argc, char **argv){

	ArgParser ap;
	ap.parseArgs(argc,argv);

	if(ap.exists(HELP) || ap.count() == 0){
		ap.menu();
		return 0;
	}

	testReadWrite(ap);
	//vz::pause();

	return 0;
}
