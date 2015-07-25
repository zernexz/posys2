#include <iostream>
#include <ctime>

#define MUL_TIMES 27648
#define ADD_TIMES 276480000
#define FP float
using namespace std;

int main(){
	
	FP* af=new FP[1280*720*3+10];
	clock_t begin_time = clock();
	cout << "init" << endl;
	
	long q=1280*720*3;
	for(long j=0;j<q;j++){
		af[j]=(FP)(j*j)/(3.0);
	}
	std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
	cout << "mul : MUL_TIMES" << endl;
	begin_time = clock();
	// do something
	
	
	
	for(long i=0,j=10000;i<MUL_TIMES;i++){
		if(j>=q)
			j=0;
		for(int d=0;d<10000;d++){
			af[j] = af[j-d] * af[j+d];
		}
	}
	
	std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
	
	
	cout << "add : ADD_TIMES" << endl;
	begin_time = clock();
	// do something
	for(long i=0,j=5;i<ADD_TIMES;i++){
		if(j>=q)
			j=0;
		af[j] = af[j-1] + af[j-2];
	}
	
	std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
}
