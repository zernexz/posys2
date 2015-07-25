#ifndef OBEAM_H
#define OBEAM_H

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/numeric/ublas/lu.hpp>

#include <glm/glm.hpp>

#include "cammat.h"
#include "invert_matrix.hpp"
#include "serial.hpp"

#include <cmath>

using glm::mat4;
using glm::vec4;
using glm::vec3;

using namespace std;
using namespace cv;



namespace CERN{

class OBeam{
private:
	float size(Point3f p){
		return sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
	}

	Point3f normalize(Point3f p){
		float ss=size(p);
		return Point3f(p.x/ss,p.y/ss,p.z/ss);
	}

public:
	OBeam()
{
	//Init
};

	vector< Point3f > merge(vector<Point3f> vps,vector<Point3f> out){
		
		for(int i=0;i<vps.size();i++){
			out.push_back( vps[i] );
		}

		return out;
	}

	vector< Point3f > collide_3dps(OBeam ob,int nd=0){
		int ret=0;
		vector< Point3f > out;
		for(unsigned int q=0;q<ob.rays.size();q++){
			int w = (q+1)%ob.rays.size();
			for(unsigned int i=0;i<rays.size();i++){
				int j=(i+1)%rays.size();

				Point3f Ia=ob.rays[q],Ib=ob.pos; //ray from another
				Point3f P0=pos,P1=rays[i],P2=rays[j]; //triangle from this
				
				boost::numeric::ublas::matrix<double> cv (3, 1);
				cv (0, 0) = Ia.x - P0.x;
				cv (1, 0) = Ia.y - P0.y;
				cv (2, 0) = Ia.z - P0.z;		
		
				boost::numeric::ublas::matrix<double> tm (3, 3);
				tm (0, 0) = Ia.x - Ib.x;
				tm (1, 0) = Ia.y - Ib.y;
				tm (2, 0) = Ia.z - Ib.z;
		
				tm (0, 1) = P1.x - P0.x;
				tm (1, 1) = P1.y - P0.y;
				tm (2, 1) = P1.z - P0.z;

				tm (0, 2) = P2.x - P0.x;
				tm (1, 2) = P2.y - P0.y;
				tm (2, 2) = P2.z - P0.z;
				
				
				float im_det = tm(0,0)*(tm(1,1)*tm(2,2)-tm(1,2)*tm(2,1))  -  tm(0,1)*(tm(1,0)*tm(2,2)-tm(1,2)*tm(2,0))  +  tm(0,2)*(tm(1,0)*tm(2,1)-tm(1,1)*tm(2,0));

				if(im_det == 0){ cout << "det = 0" << endl; continue; /*return 0;*/}
				
				boost::numeric::ublas::matrix<double> tt (3, 3);

				tt(0,0)= tm(1,1)*tm(2,2)-tm(2,1)*tm(1,2);
				tt(0,1)= tm(0,2)*tm(2,1)-tm(2,2)*tm(0,1);
				tt(0,2)= tm(0,1)*tm(1,2)-tm(1,1)*tm(0,2);
				tt(1,0)= tm(1,2)*tm(2,0)-tm(1,0)*tm(2,2);
				tt(1,1)= tm(0,0)*tm(2,2)-tm(2,0)*tm(0,2);
				tt(1,2)= tm(0,2)*tm(1,0)-tm(1,2)*tm(0,0);
				tt(2,0)= tm(1,0)*tm(2,1)-tm(2,0)*tm(1,1);
				tt(2,1)= tm(0,1)*tm(2,0)-tm(2,1)*tm(0,0);
				tt(2,2)= tm(0,0)*tm(1,1)-tm(1,0)*tm(0,1);


				tt(0,0)/=im_det;
				tt(0,1)/=im_det;
				tt(0,2)/=im_det;
				tt(1,0)/=im_det;
				tt(1,1)/=im_det;
				tt(1,2)/=im_det;
				tt(2,0)/=im_det;
				tt(2,1)/=im_det;
				tt(2,2)/=im_det;
				
				tm = tt;
				

				boost::numeric::ublas::matrix<double> rv;
				rv = boost::numeric::ublas::prod(tm, cv);
				//cout << "> " << rv(0,0) << " " << rv(1,0) << " " << rv(2,0) << endl;
				if( rv(0,0)>=0 && rv(0,0)<=1 &&
				    rv(1,0)>=0 && rv(1,0)<=1 &&
				    rv(2,0)>=0 && rv(2,0)<=1 &&
				    rv(1,0) + rv(2,0)<=1 ){
						//cout << "# " << rv(0,0) << " " << rv(1,0) << " " << rv(2,0) << endl;
						//ret++;
						out.push_back(Point3f( Ia.x + (Ib.x-Ia.x) * rv(0,0) , Ia.y + (Ib.y-Ia.y) * rv(0,0) , Ia.z + (Ib.z-Ia.z) * rv(0,0) ));	
				}
			}
		}

		if(nd==1)
			return out;

		vector<Point3f> vps = ob.collide_3dps(*this,1);
		for(int i=0;i<vps.size();i++){
			out.push_back( vps[i] );
		}

		return out;

		//if(nd==1) return ret;
		//return ( ret*ob.rays.size() + ob.collide(*this,1)*rays.size() )/(ob.rays.size()*rays.size())*16; //Reflect property
	}



	int collide(OBeam ob,int nd=0){
		int ret=0;
		for(unsigned int q=0;q<ob.rays.size();q++){
			int w = (q+1)%ob.rays.size();
			for(unsigned int i=0;i<rays.size();i++){
				int j=(i+1)%rays.size();

				Point3f Ia=ob.rays[q],Ib=ob.pos; //ray from another
				Point3f P0=pos,P1=rays[i],P2=rays[j]; //triangle from this
				
				boost::numeric::ublas::matrix<double> cv (3, 1);
				cv (0, 0) = Ia.x - P0.x;
				cv (1, 0) = Ia.y - P0.y;
				cv (2, 0) = Ia.z - P0.z;		
		
				boost::numeric::ublas::matrix<double> tm (3, 3);
				tm (0, 0) = Ia.x - Ib.x;
				tm (1, 0) = Ia.y - Ib.y;
				tm (2, 0) = Ia.z - Ib.z;
		
				tm (0, 1) = P1.x - P0.x;
				tm (1, 1) = P1.y - P0.y;
				tm (2, 1) = P1.z - P0.z;

				tm (0, 2) = P2.x - P0.x;
				tm (1, 2) = P2.y - P0.y;
				tm (2, 2) = P2.z - P0.z;
				
				
				float im_det = tm(0,0)*(tm(1,1)*tm(2,2)-tm(1,2)*tm(2,1))  -  tm(0,1)*(tm(1,0)*tm(2,2)-tm(1,2)*tm(2,0))  +  tm(0,2)*(tm(1,0)*tm(2,1)-tm(1,1)*tm(2,0));

				if(im_det == 0){ cout << "det = 0" << endl; continue; /*return 0;*/}
				
				boost::numeric::ublas::matrix<double> tt (3, 3);

				tt(0,0)= tm(1,1)*tm(2,2)-tm(2,1)*tm(1,2);
				tt(0,1)= tm(0,2)*tm(2,1)-tm(2,2)*tm(0,1);
				tt(0,2)= tm(0,1)*tm(1,2)-tm(1,1)*tm(0,2);
				tt(1,0)= tm(1,2)*tm(2,0)-tm(1,0)*tm(2,2);
				tt(1,1)= tm(0,0)*tm(2,2)-tm(2,0)*tm(0,2);
				tt(1,2)= tm(0,2)*tm(1,0)-tm(1,2)*tm(0,0);
				tt(2,0)= tm(1,0)*tm(2,1)-tm(2,0)*tm(1,1);
				tt(2,1)= tm(0,1)*tm(2,0)-tm(2,1)*tm(0,0);
				tt(2,2)= tm(0,0)*tm(1,1)-tm(1,0)*tm(0,1);


				tt(0,0)/=im_det;
				tt(0,1)/=im_det;
				tt(0,2)/=im_det;
				tt(1,0)/=im_det;
				tt(1,1)/=im_det;
				tt(1,2)/=im_det;
				tt(2,0)/=im_det;
				tt(2,1)/=im_det;
				tt(2,2)/=im_det;
				
				tm = tt;
				

				boost::numeric::ublas::matrix<double> rv;
				rv = boost::numeric::ublas::prod(tm, cv);
				//cout << "> " << rv(0,0) << " " << rv(1,0) << " " << rv(2,0) << endl;
				if( rv(0,0)>=0 && rv(0,0)<=1 &&
				    rv(1,0)>=0 && rv(1,0)<=1 &&
				    rv(2,0)>=0 && rv(2,0)<=1 &&
				    rv(1,0) + rv(2,0)<=1 ){
						//cout << "# " << rv(0,0) << " " << rv(1,0) << " " << rv(2,0) << endl;
						ret++;	
				}
			}
		}
		
		if(nd==1) return ret;
		return ( ret*ob.rays.size() + ob.collide(*this,1)*rays.size() )/(ob.rays.size()*rays.size())*16; //Reflect property
	}
	
	void set_up(Cammat cm,vector<Point2f> abl){
		set_pos(cm);
		this->cm=cm;
		rays.clear();
		for(unsigned int i=0;i<abl.size();i++){
			rays.push_back(cm.cm_223(abl[i]));
		}
		resize_rays();
	}

	void set_pos(Cammat cm){
		vector<float> cpos(3);
		for(int q=0;q<3;q++){
				cpos[q]=0;
				for(int w=0;w<3;w++){
					cpos[q]+= cm.rotT.at<double>(q,w)*cm.tvec.at<double>(w, 0);
				}
				cpos[q]*=-1;
				//cpos[q]*=scale;
		}
		pos.x=cpos[0];
		pos.y=cpos[1];
		pos.z=cpos[2];
	}

	void resize_rays(){
		float mxl=50;
		for(unsigned i=0;i<rays.size();i++){
			rays[i]= ( rays[i] - pos )*mxl + pos; 
		}
	}

	Cammat cm;
	Point3f pos;
	vector < Point3f > rays;//note : point not a rays
};

}

#endif
