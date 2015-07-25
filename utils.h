#ifndef UTILS_H
#define UTILS_H

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <random>
#include <cmath>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>

#ifndef NOT_USE
#define NOT_USE 2<<20
#endif

using namespace std;


template < typename FP >
class Utils{
private:

public:
	bool return_v;//false
	FP v_val;//0.0f
	
	Utils():return_v(false),v_val(0.0f){
	}

	FP mrand(){//~ Math.random()   [0,1]
	    // Seed with a real random value, if available
	    std::random_device rd;
	 
	    // Choose a random mean between 1 and 6
	    std::default_random_engine e1(rd());
	    std::uniform_real_distribution<FP> uniform_dist(0,1);
	    FP mean = uniform_dist(e1);
		return mean;
	}

 	FP gaussRandom(){
		if(return_v){
			return_v = false;
			return v_val;
		}
		FP u = 2*mrand()-1;
		FP v = 2*mrand()-1;
		FP r = u*u + v*v;
		if(r == 0 || r > 1) return gaussRandom();
		FP c = sqrt(-2*log(r)/r);
		v_val = v*c; // cache this
		return_v = true;
		return u*c;
	}

	FP randf(FP a,FP b){
		return mrand()*(b-a)+a;
	}

	int randi(FP a,FP b){
		return floor(mrand()*(b-a)+a);
	}

	FP randn(FP mu,FP std){
		return mu+gaussRandom()*std;
	}

	vector<FP> zeros(int n){
	    vector<FP> p;
	    for(int i=0;i<n;i++)
			p.push_back(FP(0));
		return p;
	}

	bool arrContains(const vector<FP> &arr,FP elt){
	    return std::find(arr.begin(), arr.end(), elt) != arr.end();
	}

	vector<FP> arrUnique(const vector<FP> &arr){
	    vector<FP> b;
	    for(int i=0,n=arr.size();i<n;i++) {
	      if(!arrContains(b, arr[i])) {
		b.push_back(arr[i]);
	      }
	    }
	    return b;
	}


	map<string,FP> maxmin(const vector<FP> &w){
		map<string,FP> m;
		if(w.size() == 0){ return m; }
		FP maxv = w[0];
		FP minv = w[0];
		FP maxi = 0;
		FP mini = 0;
		int n = w.size();
		for(int i=0;i<n;i++){
			if(w[i] > maxv) { maxv = w[i]; maxi = i; }
			if(w[i] < minv) { minv = w[i]; mini = i; }
		}
		m["maxi"]=maxi;
		m["maxv"]=maxv;
		m["mini"]=mini;
		m["minv"]=minv;
		m["dv"]=maxv-minv;
	}

	vector<int> randperm(int n){
		int i=n;
		int j=0;
		int tmp;
		vector<int> arr;
		for(int q=0;q<n;q++)
			arr.push_back(q);
		while(i--){
			j = floor( mrand() * (i+1) );
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
		}
		return arr;
	}

	FP weigthedSample(vector<FP> lst,vector<FP> probs){
		float p = randf(0,1);
		float cumprob = 0.0f;
		for(int k=0,n=lst.size();k<n;k++){
			cumprob += probs[k];
			if(p < cumprob){ return lst[k]; }
		}
		return FP(0);
	}
	
	
};

#endif 
