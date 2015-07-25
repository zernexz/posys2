#include "camera.h"
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>


#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>


#include <glm/glm.hpp>
using glm::mat4;
using glm::vec4;
using glm::vec3;
using namespace std;
using namespace cv;
using namespace CERN;

Camera::Camera():intrinsic(3, 3, CV_32FC1)//Init ex.  a(1) , b(50.0f)
{
	//Init
}

