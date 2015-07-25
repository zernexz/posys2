#ifndef CAMMAT_H
#define CAMMAT_H


#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>

#include <glm/glm.hpp>

 
#include "serial.hpp"

using glm::mat4;
using glm::vec4;
using glm::vec3;

using namespace std;
using namespace cv;

namespace CERN{

class Cammat{
private:
    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & rot;
        ar & tvec;
        ar & inited;
    }

public:
	Cammat()
	{
		//Init
	};

	Point2f cm_322(Point3f pt){
	
						double p[3]={0};
						double sp1[3]={0};
						for(int row=0; row<3; ++row)
						{
						
						   p[row]+=rot.at<double>(row, 0)*pt.x;
						   p[row]+=rot.at<double>(row, 1)*pt.y;
						   p[row]+=rot.at<double>(row, 2)*pt.z;
						   p[row]+=tvec.at<double>(row, 0);
						}
					
					
						p[0]/=p[2];
						p[1]/=p[2];
					
						/*
						p[0]*=0.00098;
						p[1]*=0.00098;
						*/
					
						for(int row=0; row<2; ++row)
						{
						   sp1[row]+=intrinsic.at<double>(row, 0)*p[0];
						   sp1[row]+=intrinsic.at<double>(row, 1)*p[1];
						   sp1[row]+=intrinsic.at<double>(row, 2);
						}
					
						//circle(mimg,Point(sp1[0],sp1[1]),5,Scalar((int)(255*0),(int)(255*0),(int)(255*1) ));
						//circle(mimg,Point(sp1[0],sp1[1]),5,Scalar(0,255,255 ));
						//cout << sp1[0] << " , " << sp1[1]  << "         " << p[2]<< endl;
				return Point2f(sp1[0],sp1[1]);
	}
	Point3f cm_223(Point2f p){
				double xp[3]={0};
				double xsp[3]={0};
			
				double sp1[3]={0};
			
				float abSx=0.000985; //0.000985;
				float abSy=0.00098; //0.00098;
				xsp[0]=(p.x-intrinsic.ptr<double>(0)[2])*abSx;
				xsp[1]=(p.y-intrinsic.ptr<double>(1)[2])*abSy;
				xsp[2]=1;
			
				for(unsigned int row=0; row<3; ++row)
				{
					xsp[row]-=tvec.at<double>(row, 0);
				}
			
				for(unsigned int row=0; row<3; ++row)
				{
				
				   xp[row]+=rotT.at<double>(row, 0)*xsp[0];
				   xp[row]+=rotT.at<double>(row, 1)*xsp[1];
				   xp[row]+=rotT.at<double>(row, 2)*xsp[2];
				   //xp[row]-=tvec.at<double>(row, 0);
				}
			
				//cout << xp[0] << " , " << xp[1] << " , " << xp[2] << endl;
				return Point3f(xp[0],xp[1],xp[2]);
			
				double lv[3]={0};
				lv[0]=xp[0]-tvec.at<double>(0, 0);
				lv[1]=xp[1]-tvec.at<double>(1, 0);
				lv[2]=xp[2]-tvec.at<double>(2, 0);
				double lvs = sqrt( lv[0]*lv[0] + lv[1]*lv[1] + lv[2]*lv[2] );
				double nlv[3]={0};
				nlv[0]=lv[0]/lvs;
				nlv[1]=lv[1]/lvs;
				nlv[2]=lv[2]/lvs;
			
	}



	Mat rot;
	Mat rotT;
	Mat tvec;
	Mat intrinsic;
	int inited;
};

}
#endif
