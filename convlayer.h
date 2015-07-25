#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "layer.h"
#include "vol.h"
#include "utils.h"
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
#include <ctime>

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
class ConvLayer : public Layer<FP>{
private:

public:
	int out_depth;
	int out_sx;
	int out_sy;

	int sx;
	int sy;

	int in_depth;
	int in_sx;
	int in_sy;
	
	int stride;
	int pad;
	FP l1_decay_mul;
	FP l2_decay_mul;



	string layer_type;

	FP bias;
	Vol<FP>* biases;

	vector<Vol<FP>* > filters;

	
	//Out:{d,x,y}   In:{d,x,y} Conv:{stride,pad,l1_decay,l2_decay}
	ConvLayer(int out_depth,int sx,int sy,int in_depth,int in_sx,int in_sy,int stride=1,int pad=0,FP l1_decay_mul=FP(0),FP l2_decay_mul=FP(1),FP bias_pref=FP(0)):sx(sx),sy(sy),in_depth(in_depth),in_sx(in_sx),in_sy(in_sy),stride(stride),pad(pad),l1_decay_mul(l1_decay_mul),l2_decay_mul(l2_decay_mul),layer_type("conv"),bias(bias_pref),biases(NULL){

	this->in_act=NULL;
	this->out_act=NULL;
	//cout << "cv 0" << endl;
	this->out_sx = floor( (this->in_sx + this->pad * 2 - this->sx) / this->stride + 1 );
	this->out_sy = floor( (this->in_sy + this->pad * 2 - this->sy) / this->stride + 1 );
	//cout << "cv 1" << endl;
	this->bias = bias_pref;
	//cout << "cv 2" << endl;
	this->out_depth=out_depth;
	for(int i=0;i<this->out_depth;i++){
		//cout << "cv 2.1" << endl;
		this->filters.push_back(new Vol<FP>(this->sx,this->sy,this->in_depth));//cout << "cv 2.2" << endl;
	}
	//cout << "cv 3" << endl;
	this->biases = new Vol<FP>(1, 1, this->out_depth , FP(bias) );
	//cout << "cv 4" << endl;
	}
	void dtor(){
		cout << "clearrr" << endl;
		if(this->biases != NULL){delete this->biases;this->biases=NULL;}
		//cout << "clearrr1" << endl;

		for(int i=0;i<this->filters.size();i++)
			delete this->filters[i];
		//cout << "clearrr2" << endl;
		this->filters.clear();
		//cout << "clearrr3" << endl;
		if(this->in_act != NULL){delete this->in_act;this->in_act=NULL;}
		//cout << "clearrr5" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		//cout << "clearrr4" << endl;
	}
	~ConvLayer(){
		dtor();
	}

vector<FP> get_all_w(){
	vector<FP> out;

	Vol<FP>* V;
	vector< Vol<FP>* > list;

	for(int q=0;q<this->filters.size();q++)
		list.push_back(this->filters[q]);
	list.push_back(this->biases);

	for(int z=0;z<list.size();z++){
		V=list[z];
		int size=V->w.size();
		//cout << size << endl;
		for(int q=0;q<size;q++){
			out.push_back(V->w[q]);
		}
	}

	return out;
}
void set_all_w(vector<FP> aw){
	Vol<FP>* V;
	vector< Vol<FP>* > list;
	vector<int> slist;
	int as=0;

	for(int q=0;q<this->filters.size();q++){
		list.push_back(this->filters[q]);
		slist.push_back(this->filters[q]->w.size());
		as+=this->filters[q]->w.size();		
	}
	slist.push_back(this->biases->w.size());
	list.push_back(this->biases);
	as+=this->biases->w.size();

	for(int i=0,q=0;i<slist.size();i++){
		V = list[i];
		for(int j=0;j<slist[i];j++,q++){
			V->w[j]=aw[q];
			
		}
	}

}

	Vol<FP>* forward(Vol<FP>* V,bool is_training=false){

		//cout << "feed a" << endl;

		this->in_act = V;
		
		//cout << "feed b" << endl;
		Vol<FP>* A = new Vol<FP>(this->out_sx,this->out_sy,this->out_depth,FP(0.0));
		//cout << "feed c" << endl;
		int V_sx = V->sx;
		int V_sy = V->sy;
		int V_depth=V->depth;
		int xy_stride = this->stride;
		//cout << "feed ddd" << endl;


		clock_t begin_time = clock();
		//#pragma omp parallel for
		for(int d=0;d<this->out_depth;d++){
			Vol<FP>* f = this->filters[d];
			int x = -this->pad;
			int y = -this->pad;
			int f_sx=f->sx;
			int f_depth=f->depth;
			
			for(int ay=0;ay<this->out_sy;y+=xy_stride,ay++){
				x = -this->pad;
				for(int ax=0;ax<this->out_sx;x+=xy_stride,ax++){
					FP a(0);
					for(int fy=0;fy<f->sy;fy++){
						int oy = y+fy;
						for(int fx=0;fx<f->sx;fx++){
							int ox = x+fx;
							if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx){
								for(int fd=0;fd<f->depth;fd++){
								a += f->w[((f_sx * fy)+fx)*f_depth + fd] * V->w[((V_sx * oy)+ox)*V_depth + fd];
								}
							}
						}
					}
					a += this->biases->w[d];
					A->set(ax,ay,d,a);
				}
			}
		}
		//std::cout << "ConvLayer : " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

		//cout << "feed e" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		//cout << "feed f" << endl;
		this->out_act = A;
		//cout << "feed g" << endl;
		//cout << "feed h" << endl;

		
		return this->out_act;
	}
	void backward(int tmpy=0){
		Vol<FP>* V = this->in_act;
		Utils<FP> ut;
		V->dw = ut.zeros(V->w.size());
		int V_sx = V->sx;
		int V_sy = V->sy;
		int xy_stride = this->stride;
		
		for(int d=0;d<this->out_depth;d++){
			Vol<FP>* f = this->filters[d];
			int x = -this->pad;
			int y = -this->pad;
			for(int ay=0;ay<this->out_sy;y+=xy_stride,ay++){
				x = -this->pad;
				for(int ax=0;ax<this->out_sx;x+=xy_stride,ax++){
					FP chain_grad  = this->out_act->get_grad(ax,ay,d);
					//if(abs(chain_grad)>0.001)cout << chain_grad << endl;
					for(int fy=0;fy<f->sy;fy++){
						int oy=y+fy;
						for(int fx=0;fx<f->sx;fx++){
							int ox=x+fx;
							if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx){
								for(int fd=0;fd<f->depth;fd++){
									int ix1 = ((V_sx * oy)+ox)*V->depth + fd;
									int ix2 = ((f->sx * fy)+fx)*f->depth + fd;
									f->dw[ix2] += V->w[ix1]*chain_grad;
									V->dw[ix1] += f->w[ix2]*chain_grad;
								}
							}
						}
					}
					this->biases->dw[d] += chain_grad;
				}
			}
		}
	}
	
	vector< map<string,vector<FP>* > > getParamsAndGrads(){
		vector< map<string,vector<FP>* > > v;//cout << "!1" << endl;
		/*for(int d=0;d<this->filters.size();d++){
			for(int l=0;l<this->filters[d]->w.size();l++){
				this->filters[d]->w[l]=FP(0.1);
			}
		}*/

		for(int i=0;i<this->out_depth;i++){
			map<string,vector<FP>* > m;
			m["params"] = &this->filters[i]->w; 
			/*
			vector<FP>& ww= *tt;
			for(int l=0;l<ww.size();l++){
				ww[l]=FP(0.1);
			}*/

			m["grads"] = &this->filters[i]->dw;
			v.push_back(m);
		}//cout << "!4" << endl;
		map<string,vector<FP>* > m;
		m["params"] = &this->biases->w;
		m["grads"] = &this->biases->dw;
		v.push_back(m);
		return v;
	}
string get_layer_type(){
	return this->layer_type;
}
Vol<FP>* get_in_act(){
	return this->in_act;
}
Vol<FP>* get_out_act(){
	return this->out_act;
}
	
};


/**
 * Mat::data Specification
 * 2x2  1 channel
 * [ R , R ;
 *   R , R ]
 * 
 * 2x2  2 channel
 * [ R , G , R , G ;
 *   R , G , R , G ]
 * 
 * 2x2  3 channel
 * [ R , G , B , R , G , B ;
 *   R , G , B . R , G , B ] 
 */
#endif
