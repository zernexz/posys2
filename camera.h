#ifndef CAMERA_H
#define CAMERA_H


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

class Camera{
private:
    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & object_points;
        ar & image_points;
        ar & image;
        ar & intrinsic;
        ar & distCoeffs;
        ar & rvecs;
        ar & tvecs;
    }

public:
	Camera();
	vector<vector<Point3f> > object_points;
    	vector<vector<Point2f> > image_points;
	Mat image;
	Mat intrinsic;
    	Mat distCoeffs;
    	vector<Mat> rvecs;
    	vector<Mat> tvecs;

};

}
#endif
