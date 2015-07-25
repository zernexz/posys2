// Yannick Verdie 2010
// --- Please read help() below: ---

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <dirent.h>

#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "picojson/picojson.h"

#include "camera.h"
#include "memdump.h"

#include "utils.h"
#include "vol.h"

#include "inputlayer.h"
#include "convlayer.h"
#include "sigmoidlayer.h"
#include "poollayer.h"
#include "relulayer.h"
#include "fullyconnlayer.h"
#include "softmaxlayer.h"
#include "convnet.h"

#include "cammat.h"

#include "obeam.h"

#include "invert_matrix.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/numeric/ublas/lu.hpp>


#define FOCAL_LENGTH 600
#define CUBE_SIZE 0.5

#define CAMERA_INTR 0
#define CAMERA_DIST 1

#define CAMERA -1


using namespace std;
using namespace cv;
using namespace CERN;


 
typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef set<int> si;
typedef map<string, int> msi;

// To simplify repetitions/loops, Note: define your loop style and stick with it!
#define REP(i, a, b) \
for (int i = int(a); i <= int(b); i++) // a to b, and variable i is local!
#define TRvi(c, it) \
for (vi::iterator it = (c).begin(); it != (c).end(); it++)
#define TRvii(c, it) \
for (vii::iterator it = (c).begin(); it != (c).end(); it++)
#define TRmsi(c, it) \
for (msi::iterator it = (c).begin(); it != (c).end(); it++)


float posePOSIT[] = { 1, 0, 0, 0,
								 0, 1, 0, 0,
								 0, 0, 1, 0,
								 0, 0, 0, 1 };




static void help()
{
    cout << "This demo demonstrates the use of the Qt enhanced version of the highgui GUI interface\n"
            "and dang if it doesn't throw in the use of of the POSIT 3D tracking algorithm too\n"
            "It works off of the video: cube4.avi\n"
            "Using OpenCV version " << CV_VERSION << "\n\n"

            " 1) This demo is mainly based on work from Javier Barandiaran Martirena\n"
            "    See this page http://code.opencv.org/projects/opencv/wiki/Posit.\n"
            " 2) This is a demo to illustrate how to use **OpenGL Callback**.\n"
            " 3) You need Qt binding to compile this sample with OpenGL support enabled.\n"
            " 4) The features' detection is very basic and could highly be improved\n"
            "    (basic thresholding tuned for the specific video) but 2).\n"
            " 5) Thanks to Google Summer of Code 2010 for supporting this work!\n" << endl;
}

static char* str_concat(const char str[512],int d){
	char numstr[512];
	sprintf(numstr,"%s%d",str,d);
	return numstr;
} 



static void renderCube(float size)
{
// White side - BACK
glBegin(GL_POLYGON);
glColor3f(   1.0,  1.0, 1.0 );
glVertex3f(  0.5, -0.5, 0.5 );
glVertex3f(  0.5,  0.5, 0.5 );
glVertex3f( -0.5,  0.5, 0.5 );
glVertex3f( -0.5, -0.5, 0.5 );
glEnd();
 
// Purple side - RIGHT
glBegin(GL_POLYGON);
glColor3f(  1.0,  0.0,  1.0 );
glVertex3f( 0.5, -0.5, -0.5 );
glVertex3f( 0.5,  0.5, -0.5 );
glVertex3f( 0.5,  0.5,  0.5 );
glVertex3f( 0.5, -0.5,  0.5 );
glEnd();
 
// Green side - LEFT
glBegin(GL_POLYGON);
glColor3f(   0.0,  1.0,  0.0 );
glVertex3f( -0.5, -0.5,  0.5 );
glVertex3f( -0.5,  0.5,  0.5 );
glVertex3f( -0.5,  0.5, -0.5 );
glVertex3f( -0.5, -0.5, -0.5 );
glEnd();
 
// Blue side - TOP
glBegin(GL_POLYGON);
glColor3f(   0.0,  0.0,  1.0 );
glVertex3f(  0.5,  0.5,  0.5 );
glVertex3f(  0.5,  0.5, -0.5 );
glVertex3f( -0.5,  0.5, -0.5 );
glVertex3f( -0.5,  0.5,  0.5 );
glEnd();
 
// Red side - BOTTOM
glBegin(GL_POLYGON);
glColor3f(   1.0,  0.0,  0.0 );
glVertex3f(  0.5, -0.5, -0.5 );
glVertex3f(  0.5, -0.5,  0.5 );
glVertex3f( -0.5, -0.5,  0.5 );
glVertex3f( -0.5, -0.5, -0.5 );
glEnd();
/*
    glBegin(GL_QUADS);
    // Front Face
    glNormal3f( 0.0f, 0.0f, 1.0f);
    glVertex3f( 0.0f,  0.0f,  0.0f);
    glVertex3f( size,  0.0f,  0.0f);
    glVertex3f( size,  size,  0.0f);
    glVertex3f( 0.0f,  size,  0.0f);
    // Back Face
    glNormal3f( 0.0f, 0.0f,-1.0f);
    glVertex3f( 0.0f,  0.0f, size);
    glVertex3f( 0.0f,  size, size);
    glVertex3f( size,  size, size);
    glVertex3f( size,  0.0f, size);
    // Top Face
    glNormal3f( 0.0f, 1.0f, 0.0f);
    glVertex3f( 0.0f,  size,  0.0f);
    glVertex3f( size,  size,  0.0f);
    glVertex3f( size,  size, size);
    glVertex3f( 0.0f,  size, size);
    // Bottom Face
    glNormal3f( 0.0f,-1.0f, 0.0f);
    glVertex3f( 0.0f,  0.0f,  0.0f);
    glVertex3f( 0.0f,  0.0f, size);
    glVertex3f( size,  0.0f, size);
    glVertex3f( size,  0.0f,  0.0f);
    // Right face
    glNormal3f( 1.0f, 0.0f, 0.0f);
    glVertex3f( size,  0.0f, 0.0f);
    glVertex3f( size,  0.0f, size);
    glVertex3f( size,  size, size);
    glVertex3f( size,  size, 0.0f);
    // Left Face
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f( 0.0f,  0.0f, 0.0f);
    glVertex3f( 0.0f,  size, 0.0f);
    glVertex3f( 0.0f,  size, size);
    glVertex3f( 0.0f,  0.0f, size);
    glEnd();*/
    
}

static void on_opengl(void* param)
{
    //Draw the object with the estimated pose
    glLoadIdentity();
    glScalef( 1.0f, 1.0f, -1.0f);
    glMultMatrixf( posePOSIT );
    glEnable( GL_LIGHTING );
    glEnable( GL_LIGHT0 );
    glEnable( GL_BLEND );
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    renderCube( 1 );
    glDisable(GL_BLEND);
    glDisable( GL_LIGHTING );
}

static void initPOSIT(std::vector<CvPoint3D32f> * modelPoints)
{
    // Create the model pointss
    modelPoints->push_back(cvPoint3D32f(0.0f, 0.0f, 0.0f)); // The first must be (0, 0, 0)
    modelPoints->push_back(cvPoint3D32f(0.0f, 0.0f, CUBE_SIZE));
    modelPoints->push_back(cvPoint3D32f(CUBE_SIZE, 0.0f, 0.0f));
    modelPoints->push_back(cvPoint3D32f(0.0f, CUBE_SIZE, 0.0f));
}

static void foundCorners(vector<CvPoint2D32f> * srcImagePoints, const Mat & source, Mat & grayImage)
{
    cvtColor(source, grayImage, COLOR_RGB2GRAY);
    GaussianBlur(grayImage, grayImage, Size(11, 11), 0, 0);
    normalize(grayImage, grayImage, 0, 255, NORM_MINMAX);
    threshold(grayImage, grayImage, 26, 255, THRESH_BINARY_INV); //25

    Mat MgrayImage = grayImage;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(MgrayImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    Point p;
    vector<CvPoint2D32f> srcImagePoints_temp(4, cvPoint2D32f(0, 0));

    if (contours.size() == srcImagePoints_temp.size())
    {
        for (size_t i = 0; i < contours.size(); i++ )
        {
            p.x = p.y = 0;

            for (size_t j = 0 ; j < contours[i].size(); j++)
                p += contours[i][j];

            srcImagePoints_temp.at(i) = cvPoint2D32f(float(p.x) / contours[i].size(), float(p.y) / contours[i].size());
        }

        // Need to keep the same order
        // > y = 0
        // > x = 1
        // < x = 2
        // < y = 3

        // get point 0;
        size_t index = 0;
        for (size_t i = 1 ; i<srcImagePoints_temp.size(); i++)
            if (srcImagePoints_temp.at(i).y > srcImagePoints_temp.at(index).y)
                index = i;
        srcImagePoints->at(0) = srcImagePoints_temp.at(index);

        // get point 1;
        index = 0;
        for (size_t i = 1 ; i<srcImagePoints_temp.size(); i++)
            if (srcImagePoints_temp.at(i).x > srcImagePoints_temp.at(index).x)
                index = i;
        srcImagePoints->at(1) = srcImagePoints_temp.at(index);

        // get point 2;
        index = 0;
        for (size_t i = 1 ; i<srcImagePoints_temp.size(); i++)
            if (srcImagePoints_temp.at(i).x < srcImagePoints_temp.at(index).x)
                index = i;
        srcImagePoints->at(2) = srcImagePoints_temp.at(index);

        // get point 3;
        index = 0;
        for (size_t i = 1 ; i<srcImagePoints_temp.size(); i++ )
            if (srcImagePoints_temp.at(i).y < srcImagePoints_temp.at(index).y)
                index = i;
        srcImagePoints->at(3) = srcImagePoints_temp.at(index);

        Mat Msource = source;
        stringstream ss;
        for (size_t i = 0; i<srcImagePoints_temp.size(); i++ )
        {
            ss << i;
            circle(Msource, srcImagePoints->at(i), 5, Scalar(0, 0, 255));
            putText(Msource, ss.str(), srcImagePoints->at(i), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
            ss.str("");

            // new coordinate system in the middle of the frame and reversed (camera coordinate system)
            srcImagePoints->at(i) = cvPoint2D32f(srcImagePoints_temp.at(i).x - source.cols / 2,
                                                 source.rows / 2 - srcImagePoints_temp.at(i).y);
        }
    }
}

static void createOpenGLMatrixFrom(float * posePOSIT, const CvMatr32f & rotationMatrix,
                                   const CvVect32f & translationVector)
{
    // coordinate system returned is relative to the first 3D input point
    for (int f = 0; f < 3; f++)
        for (int c = 0; c < 3; c++)
            posePOSIT[c * 4 + f] = rotationMatrix[f * 3 + c]; // transposed

    posePOSIT[3] = translationVector[0];
    posePOSIT[7] = translationVector[1];
    posePOSIT[11] = translationVector[2];
    posePOSIT[12] = 0.0f;
    posePOSIT[13] = 0.0f;
    posePOSIT[14] = 0.0f;
    posePOSIT[15] = 1.0f;
}


static void printMat(const Mat& M){	
	
	for(int i=0;i<M.rows;i++){
		const float* Mi = M.ptr<float>(i);
		for(int j=0;j<M.cols;j++){
			cout << "(" << i << "," << j << ") : " << Mi[j] << " ";
		}
		cout << endl;
	}
	
}


bool write_file_binary (std::string const & filename, 
  char const * data, size_t const bytes)
{
  std::ofstream b_stream(filename.c_str(), 
    std::fstream::out | std::fstream::binary);
  if (b_stream)
  {
    b_stream.write(data, bytes);
    return (b_stream.good());
  }
  return false;
}


char* read_file_binary (std::string const & filename, size_t const bytes){
	char* data=new char[bytes];
	
	std::ifstream b_stream(filename.c_str(), 
    std::fstream::out | std::fstream::binary);
  if (b_stream)
  {
    b_stream.read(data, bytes);
    return data;
  }
  return NULL;
}

static void writeMat(const char * path ,const Mat& M){
	
	long size=M.rows*M.cols;
	float * buffer = new float[size];
	
	for(int i=0;i<M.rows;i++){
		const float* Mi = M.ptr<float>(i);
		for(int j=0;j<M.cols;j++){
			buffer[i*M.cols + j]=Mi[j];
		}
		cout << endl;
	}
	
	write_file_binary(path, 
    reinterpret_cast<char const *>(buffer), 
    sizeof(float)*size);
    
	delete[] buffer;
}

static void readMat(const char * path ,Mat& M){
	
	long size=M.rows*M.cols;
	float * buffer = reinterpret_cast<float *>(read_file_binary(path,sizeof(float)*size));
	
	for(int i=0;i<M.rows;i++){
		float* Mi = M.ptr<float>(i);
		for(int j=0;j<M.cols;j++){
			Mi[j]=buffer[i*M.cols + j];
		}
		cout << endl;
	}
    
	delete[] buffer;
}

static char * getConfigPath(int type,int no){
	char winstr[512]="/home/ryouma/opencv-2.4.9/samples/cpp/posys/camera_config/";
	char numstr[512]; // enough to hold all numbers up to 64-bits
    sprintf(numstr,"conf_%d_%d.bin",type,no);
    strcat( winstr ,numstr);
    return winstr;
}

static char * getTrainingSetPath(int type,int no){
	return "/home/ryouma/Desktop/cow.png";
	
	char winstr[512]="/home/ryouma/opencv-2.4.9/samples/cpp/posys/";
	char numstr[512]; // enough to hold all numbers up to 64-bits
    sprintf(numstr,"conf_%d_%d.bin",type,no);
    strcat( winstr ,numstr);
    return winstr;
}


static void calibrateCamera(int CamNum,int abCamNum){
	int numBoards = 40;
    int numCornersHor = 4;
    int numCornersVer = 3;
    
    printf("Enter number of corners along width: ");
    scanf("%d", &numCornersHor);

    printf("Enter number of corners along height: ");
    scanf("%d", &numCornersVer);

    printf("Enter number of boards: ");
    scanf("%d", &numBoards);
    
    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor, numCornersVer);
    VideoCapture capture = VideoCapture(CamNum);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    
    Camera cam;
    
    vector<Point2f> corners;
    int successes=0;
    
    Mat gray_image;
    capture >> cam.image;
    
    vector<Point3f> obj;
    for(int j=0;j<numSquares;j++)
        obj.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f));
        
    while(successes<numBoards)
    {
		cvtColor(cam.image, gray_image, CV_BGR2GRAY);
		bool found = findChessboardCorners(cam.image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if(found)
        {
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray_image, board_sz, corners, found);
        }
        imshow("win1", cam.image);
        
        char winstr[21]="win2"; // enough to hold all numbers up to 64-bits
        
        //char numstr[21]; // enough to hold all numbers up to 64-bits
        //sprintf(numstr,"%d",successes);
		//strcat( winstr ,numstr);

        imshow( winstr , gray_image);

        capture >> cam.image;
        int key = waitKey(1);
        
        if(key==27)

            return ;

        if(key==' ' && found!=0)
        {
            cam.image_points.push_back(corners);
            cam.object_points.push_back(obj);

            cout << "Snap stored! pic" << successes+1 << "/" << numBoards << endl;

            successes++;

            if(successes>=numBoards)
                break;
        }
	}
	//capture.release();
	
    
    cam.intrinsic.ptr<float>(0)[0] = 1;
    cam.intrinsic.ptr<float>(1)[1] = 1;
    
    
    
    
    
    calibrateCamera(cam.object_points, cam.image_points, cam.image.size(), cam.intrinsic, cam.distCoeffs, cam.rvecs, cam.tvecs);
    
    {
		std::ofstream ofs(getConfigPath(CAMERA,abCamNum));
        boost::archive::text_oarchive oa(ofs);
        oa << cam;
	}
	
    cout << cam.distCoeffs.channels() << endl;
    cout << cam.intrinsic.channels() << endl;
    
    //Build DistCoeffs & Intrinsic JSON
    
    cout << "Intrinsic" << endl;
	printMat(cam.intrinsic);
	writeMat(getConfigPath(CAMERA_INTR,abCamNum),cam.intrinsic);
	
	cout << "DistCoeffs" << endl;
	printMat(cam.distCoeffs);
	writeMat(getConfigPath(CAMERA_DIST,abCamNum),cam.distCoeffs);
	
	
	
	
	cout << "Intrinsic" << endl;
	readMat(getConfigPath(CAMERA_INTR,abCamNum),cam.intrinsic);
	printMat(cam.intrinsic);
	
	cout << "DistCoeffs" << endl;
	readMat(getConfigPath(CAMERA_DIST,abCamNum),cam.distCoeffs);
    printMat(cam.distCoeffs);
    
    /*
    Mat imageUndistorted;
    while(1)
    {
        capture >> cam.image;
        undistort(cam.image, imageUndistorted, cam.intrinsic,cam.distCoeffs);

        imshow("win1", cam.image);
        imshow("win2", imageUndistorted);
        int key = waitKey(1);
        
        if(key==27){
			capture.release();
			return ;
		}
    }*/
	 
	
	//Write JSON Camera Config to File
	
	
    
}

static void unDist(int CamNum,int abCamNum){
	VideoCapture capture = VideoCapture(CamNum);
	 
    
	Camera cam;
	
    {
        // create and open an archive for input
        std::ifstream ifs(getConfigPath(CAMERA,abCamNum));
        boost::archive::text_iarchive ia(ifs);
        // read class state from archive
        ia >> cam;
        // archive and stream closed when destructors are called
    }
    
    //calibrateCamera(cam.object_points, cam.image_points, cam.image.size(), cam.intrinsic, cam.distCoeffs, cam.rvecs, cam.tvecs);
    Mat image;
    Mat imageUndistorted;
    while(1)
    {
        capture >> image;
        undistort(image, imageUndistorted, cam.intrinsic, cam.distCoeffs);

        imshow("win1", cam.image);
        imshow("win2", imageUndistorted);
        int key = waitKey(1);
        if(key==27){
			capture.release();
			return ;
		}
    }
	 
	capture.release();
}

typedef double FP;


vector<Mat> load_cifar10(){
	
	
    vector<Mat> vm;
    for(int no;no<51;no++){
		char numstr[512]; // enough to hold all numbers up to 64-bits
		sprintf(numstr,"cifar10/cifar10_batch_%d.png",no);
		Mat mnist = imread(numstr, CV_LOAD_IMAGE_COLOR);
		for(int i=0;i<mnist.rows;i++){
			Mat m(32,32, CV_8UC3);
			for(int j=0;j<32;j++){
				for(int k=0;k<32;k++){
					for(int d=0;d<3;d++){
						
						m.data[(j*32+k)*3+d]= (uint8_t) mnist.data[((i*1024)+(j*32+k))*3+d];
						
					}
				}
			}
			//resize(m,m,Size(36,36), 0, 0, INTER_AREA);
			vm.push_back(m);
		}
	}
	
	return vm;
}

vector<int> load_cifar10_label(){
	string line;
	int val;
	vector<int> vl;
	ifstream myfile ("cifar10/label");
	  if (myfile.is_open())
	  {
		while ( getline (myfile,line,',') )
		{
			if(stringstream(line)>>val){
				vl.push_back(val);
			}
		}
		myfile.close();
	  }
	  return vl;
}


static std::vector<char> ReadAllBytes(char const* filename)
{
    ifstream ifs(filename, ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    std::vector<char>  result(pos);

    ifs.seekg(0, ios::beg);
    ifs.read(&result[0], pos);

    return result;
}

vector<int> load_cifar100_label(){
	vector<char> ab1=ReadAllBytes("cifar100/train.bin");
	vector<char> ab2=ReadAllBytes("cifar100/test.bin");
	cout << "Train Vec Size : " << ab1.size() << endl;
	cout << "Train num pic : " << ab1.size()/3074 << endl;
	cout << "Test Vec Size : " << ab2.size() << endl;
	cout << "Test num pic : " << ab2.size()/3074 << endl;
	
	vector<char> abs;
	for(unsigned int i=0;i<ab1.size();i++)
		abs.push_back(ab1[i]);
		
	for(unsigned int i=0;i<ab2.size();i++)
		abs.push_back(ab2[i]);
		
	//LabelC LabelF 1024R 1024G 1024B
	vector<Mat> vm;
	vector<int> vlc;
	vector<int> vlf;
	int len=abs.size()/3074;
	int mn=999;
	int mx=0;
		for(int i=0,pos=0;i<len;i++){
			uint8_t val;
			vlc.push_back(abs[pos++]);//0-19
			vlf.push_back(val=abs[pos++]);//0-99
			mn=(val<mn)?val:mn;
			mx=(val>mx)?val:mx;
			Mat m(32,32, CV_8UC3);
			for(int d=0;d<3;d++){
				for(int j=0;j<32;j++){
					for(int k=0;k<32;k++){
						uint8_t val = (uint8_t) abs[pos++];
						m.data[(j*32+k)*3+d]= (uint8_t) val;
						
						if(val<0)
						cout << val << endl;
					}
				}
			}
			//resize(m,m,Size(36,36), 0, 0, INTER_AREA);
			vm.push_back(m);
		}
	cout << mn << " " << mx << endl;
    cout << "End!!" << endl;		
	return vlc;
}
vector<Mat> load_cifar100(){
	vector<char> ab1=ReadAllBytes("cifar100/train.bin");
	vector<char> ab2=ReadAllBytes("cifar100/test.bin");
	cout << "Train Vec Size : " << ab1.size() << endl;
	cout << "Train num pic : " << ab1.size()/3074 << endl;
	cout << "Test Vec Size : " << ab2.size() << endl;
	cout << "Test num pic : " << ab2.size()/3074 << endl;
	
	vector<char> abs;
	for(unsigned int i=0;i<ab1.size();i++)
		abs.push_back(ab1[i]);
		
	for(unsigned int i=0;i<ab2.size();i++)
		abs.push_back(ab2[i]);
		
	//LabelC LabelF 1024R 1024G 1024B
	vector<Mat> vm;
	vector<int> vlc;
	vector<int> vlf;
	int len=abs.size()/3074;
	int mn=999;
	int mx=0;
		for(int i=0,pos=0;i<len;i++){
			uint8_t val;
			vlc.push_back(abs[pos++]);//0-19
			vlf.push_back(val=abs[pos++]);//0-99
			mn=(val<mn)?val:mn;
			mx=(val>mx)?val:mx;
			Mat m(32,32, CV_8UC3);
			for(int d=2;d>=0;d--){
				for(int j=0;j<32;j++){
					for(int k=0;k<32;k++){
						uint8_t val = (uint8_t) abs[pos++];
						m.data[(j*32+k)*3+d]= (uint8_t) val;
						
						if(val<0)
						cout << val << endl;
					}
				}
			}
			//resize(m,m,Size(36,36), 0, 0, INTER_AREA);
			vm.push_back(m);
		}
	cout << mn << " " << mx << endl;
    cout << "End!!" << endl;		
	return vm;
}

vector<Mat> load_testimage(){
  string path="test_image/";
  vector<Mat> vm;
	DIR *dirStr = NULL; 
    dirStr = opendir(path.c_str());
    dirent *nextFile = NULL;
    vector<string> listfile;
    while ((nextFile = readdir(dirStr))!=NULL)
    {
        // Avoid hidden files
        //Scan all file an dictionary
        if (nextFile->d_name[0] != '.')
        {
                    string fn(nextFile->d_name);
					listfile.push_back(fn);
        }
    }
    
    for(unsigned int i=0;i<listfile.size();i++){
		char numstr[512]; // enough to hold all numbers up to 64-bits
		sprintf(numstr,"%s%s",path.c_str(),listfile[i].c_str());
		Mat m = imread(numstr, CV_LOAD_IMAGE_COLOR);
		while(m.cols == 0) {
			 cout << "Error reading file " << numstr << endl;
			 m = imread(numstr, CV_LOAD_IMAGE_COLOR);
		}
		resize(m,m,Size((int)((float)m.rows*500.0f/m.cols),500), 0, 0);//, INTER_NEAREST);
		
		
int top, bottom, left, right;
int borderType;
Scalar value;
RNG rng(12345);
float pct=0.30;
top = (int) (pct*m.rows); bottom = (int) (pct*m.rows);
left = (int) (pct*m.cols); right = (int) (pct*m.cols);
		/*
		 borderType = BORDER_CONSTANT;
      value = Scalar( 0,0,0 );
      copyMakeBorder(m,m, top, bottom, left, right, borderType, value );
		*/
		vm.push_back(m);
		cout << numstr << endl;
	}
	
	//for(;;);
	return vm;
}

vector<Mat> do_rotate(vector<Mat> ivm){
vector<Mat> vm;
 
 for(unsigned int i=0;i<ivm.size();i++){
	 cout << "do rotate " << i << endl;
	Mat o1;
	Mat r;
	//vm.push_back(ivm[i]);
	float h=0;
	Point2f pc(ivm[i].cols/2., ivm[i].rows/2.);
	
	h=0;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());
	
	/*h=5;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());
	
	h=10;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());
	*/
	/*
	h=5;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());
	
	h=20;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());
	
	h=-5;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());
	
	h=-10;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());
	
	h=-5;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());*/
	/*
	h=-20;
	r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(ivm[i],o1, r, ivm[i].size());
	vm.push_back(o1.clone());
	*/
 }
 
 return vm;
}

vector<Mat> do_pyramid(vector<Mat> ivm){
 vector<Mat> vm;
 
 for(unsigned int i=0;i<ivm.size();i++){
	 cout << "do pyramid " << i << endl;
	Mat o1;
	float h=0;
	//vm.push_back(ivm[i]);
	
	//h=ivm[i].cols*5/6;
	//resize(ivm[i],o1,Size((int)((float)ivm[i].rows*h/ivm[i].cols),(int)h), 0, 0);//, INTER_NEAREST);
	//vm.push_back(o1.clone());
	
	/*
	h=64;
	resize(ivm[i],o1,Size( (int)h , (int)((float)ivm[i].cols*h/ivm[i].rows)  ), 0, 0);//, INTER_NEAREST);
	vm.push_back(o1.clone());
	*/
	h=256;
	resize(ivm[i],o1,Size( (int)((float)ivm[i].rows*h/ivm[i].cols) , (int)h  ), 0, 0);//, INTER_NEAREST);
	vm.push_back(o1.clone());
	
	
	h=150;
	resize(ivm[i],o1,Size( (int)((float)ivm[i].rows*h/ivm[i].cols) , (int)h  ), 0, 0);//, INTER_NEAREST);
	vm.push_back(o1.clone());
	
	h=64;
	resize(ivm[i],o1,Size( (int)((float)ivm[i].rows*h/ivm[i].cols) , (int)h  ), 0, 0);//, INTER_NEAREST);
	vm.push_back(o1.clone());
	
 }
 
 return vm;
}

void do_conv(vector<ConvNet<FP>*> cnet,vector<Mat> ivm){
 int stepx=8;
 int stepy=8;
 int winw=32;
 int winh=32;
 int nsize=3;//3 rotate  5 size
 int nclass=20;
 
 vector<float> angles;
 angles.push_back(0);
 //angles.push_back(5);
 //angles.push_back(-5);
 
 vector<string> lbs;
	//'airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
 /*lbs.push_back("airplane");
 lbs.push_back("car");
 lbs.push_back("bird");
 lbs.push_back("cat");
 lbs.push_back("deer");
 lbs.push_back("dog");
 lbs.push_back("frog");
 lbs.push_back("horse");
 lbs.push_back("ship");
 lbs.push_back("truck");*/
 
lbs.push_back("aquatic_mammals");
lbs.push_back("fish");
lbs.push_back("flowers");
lbs.push_back("food_containers");
lbs.push_back("fruit_and_vegetables");
lbs.push_back("household_electrical_devices");
lbs.push_back("household_furniture");
lbs.push_back("insects");
lbs.push_back("large_carnivores");
lbs.push_back("large_man-made_outdoor_things");
lbs.push_back("large_natural_outdoor_scenes");
lbs.push_back("large_omnivores_and_herbivores");
lbs.push_back("medium_mammals");
lbs.push_back("non-insect_invertebrates");
lbs.push_back("people");
lbs.push_back("reptiles");
lbs.push_back("small_mammals");
lbs.push_back("trees");
lbs.push_back("vehicles_1");
lbs.push_back("vehicles_2");
 
vector<Mat> ac_oim;
vector<Mat> ac_oai;
 
 for(unsigned int i=0;i<ivm.size();i++){
	Mat img=ivm[i];
	
	int stepx=img.rows/32*2;
	int stepy=stepx;
	
	vector<Mat> oim;
	vector<Mat> oai;
	
	if(i%nsize==0){//5 size
		ac_oim.clear();
		ac_oai.clear();
		for(int x=0;x<nclass;x++){
			ac_oim.push_back(img.clone());
			ac_oai.push_back(Mat::zeros(img.rows,img.cols, CV_8UC3));
		}
	}
	
	for(int x=0;x<nclass;x++){
		Mat om=img.clone();
		oim.push_back(om);
		oai.push_back(Mat::zeros(om.rows,om.cols, CV_8UC3));
	}
	cout << "do conv " << i << endl;
	
	
	for(int q=0;q<img.rows-winh;q+=stepy){
		cout << "do conv " << i << " line " << q << endl;
		for(int w=0;w<img.cols-winw;w+=stepx){
			int x=w;
			int cx=x+winw/2;
			
			int y=q;
			int cy=y+winh/2;
			
			//int mx=w+winw-1;
			//int my=w+winh-1;
			Rect myROI(x,y,32,32);
			Mat ci = img(myROI);
			
			Vol<FP>* v4 = Vol<FP>::mat_to_vol(ci);
			int pd = 0;
			FP pb = FP(0);
			
			vector<int> vote;
			vector<float> votepb;
			for(int x=0;x<nclass;x++){
				vote.push_back(0);
				votepb.push_back(FP(0));
			}
			
			#pragma omp parallel for
			for(int t=0;t<cnet.size();t++){
				cnet[t]->forward(v4);
				int tpd= cnet[t]->getPrediction();
				FP tpb = cnet[t]->getProb();
				
				vote[tpd]++;
				votepb[tpd]+=tpb;
			}
			#pragma omp barrier
			delete v4;
			
			float maxv=0;
			
			for(int x=0;x<nclass;x++){
				if(vote[x]>cnet.size()/2){
					pd=x;
					pb=votepb[x]/vote[x];
					//cout << pd << " " << pb << endl;
				}
			}
			
			
			
			if(pb<0.8)
				continue;
				
			//cout << pd << endl;
			
			//bgr
		    int dense=(int)(255.0f*pb/(2*nsize*((winw/stepx)+1))); //5*  5size
			/*
			for(int py=y;py<y+winh;py++){
				for(int px=x;px<x+winw;px++){
					oai[pd].data[(py*img.cols+px)*3+0]+=0;
					oai[pd].data[(py*img.cols+px)*3+1]+=0;
					
					if(oai[pd].data[(py*img.cols+px)*3+2]+dense>255)
						oai[pd].data[(py*img.cols+px)*3+2]=255;
					else
						oai[pd].data[(py*img.cols+px)*3+2]+=dense;
					
				}
			}*/
			
			#pragma omp parallel for
			for(int py=y;py<y+winh;py++){
				for(int px=x;px<x+winw;px++){
					oai[pd].data[(py*img.cols+px)*3+0]+=0;
					oai[pd].data[(py*img.cols+px)*3+1]+=0;
					
					if(oai[pd].data[(py*img.cols+px)*3+2]+dense>255)
						oai[pd].data[(py*img.cols+px)*3+2]=255;
					else
						oai[pd].data[(py*img.cols+px)*3+2]+=dense;
					
				}
			}
			#pragma omp barrier
			
		}

	}
	
	/*cout << "save out image " << i << endl;
	for(int x=0;x<10;x++){
		char numstr[512]; // enough to hold all numbers up to 64-bits
		sprintf(numstr,"out_image/%d_%s.png",i,lbs[x].c_str());
		imwrite( numstr, (oim[x]/2)+oai[x] );
		//sprintf(numstr,"out_image/%d_%s_dense.png",i,lbs[x].c_str());
		//imwrite( numstr, oai[x] );
	}*/
	
	if(i%nsize<=nsize-1){//5 size
		//ac_oim = img.clone();
		//ac_oai = Mat::zeros(ac_oim.rows,ac_oim.cols, CV_8UC3);
		for(int x=0;x<nclass;x++){
				Mat m=oai[x].clone();
				//cout<<"#"<<endl;
	float h=-angles[i%angles.size()];
	Point2f pc(m.cols/2., m.rows/2.);
	Mat r = getRotationMatrix2D(pc, h, 1.0);
	warpAffine(m,m, r, m.size());
	
				resize(m,m,Size(ac_oai[x].cols,ac_oai[x].rows), 0, 0);//, INTER_NEAREST);
				//cout<<"$"<<endl;
				ac_oai[x] += m;
				//cout<<"%"<<endl;
		}
	}
	
	if(i%nsize==nsize-1){//5 size
		//cout<<"^^^"<<endl;
		//ac_oim = img.clone();
		//ac_oai = Mat::zeros(ac_oim.rows,ac_oim.cols, CV_8UC3);
		for(int x=0;x<nclass;x++){
			char numstr[512]; // enough to hold all numbers up to 64-bits
			sprintf(numstr,"out_image/%d_%s.png",(i/nsize),lbs[x].c_str());
			imwrite( numstr, (ac_oim[x]/2)+ac_oai[x] );
		}
	}
 }
}





float distance(float x1,float y1,float x2,float y2){
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

Point normal(Point a){
	float d=distance(0,0,a.x,a.y);
	a.x*=1000;
	a.y*=1000;
	a.x/=d;
	a.y/=d;
	return a;
}

float dot(Point a,Point b){
	float d= a.x*b.x+a.y*b.y;
	return d;
}

float cross2d(Point a,Point b){
	float d= a.x*b.y-a.y*b.x;
	return d;
}



cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

    if (float d = ((float)(x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4)))
    {
        cv::Point2f pt;
        pt.x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
        pt.y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
        return pt;
    }
    else
        return cv::Point2f(-1, -1);
}
Point2f cm_322(Cammat cm,Point3f pt){
	
					double p[3]={0};
					double sp1[3]={0};
					for(int row=0; row<3; ++row)
					{
						
					   p[row]+=cm.rot.at<double>(row, 0)*pt.x;
					   p[row]+=cm.rot.at<double>(row, 1)*pt.y;
					   p[row]+=cm.rot.at<double>(row, 2)*pt.z;
					   p[row]+=cm.tvec.at<double>(row, 0);
					}
					
					
					p[0]/=p[2];
					p[1]/=p[2];
					
					/*
					p[0]*=0.00098;
					p[1]*=0.00098;
					*/
					
					for(int row=0; row<2; ++row)
					{
					   sp1[row]+=cm.intrinsic.at<double>(row, 0)*p[0];
					   sp1[row]+=cm.intrinsic.at<double>(row, 1)*p[1];
					   sp1[row]+=cm.intrinsic.at<double>(row, 2);
					}
					
					//circle(mimg,Point(sp1[0],sp1[1]),5,Scalar((int)(255*0),(int)(255*0),(int)(255*1) ));
					//circle(mimg,Point(sp1[0],sp1[1]),5,Scalar(0,255,255 ));
					//cout << sp1[0] << " , " << sp1[1]  << "         " << p[2]<< endl;
			return Point2f(sp1[0],sp1[1]);
}
Point3f cm_223(Cammat cm,Point2f p){
			double xp[3]={0};
			double xsp[3]={0};
			
			double sp1[3]={0};
			
			float abSx=0.000985; //0.000985;
			float abSy=0.00098; //0.00098;
			xsp[0]=(p.x-cm.intrinsic.ptr<double>(0)[2])*abSx;
			xsp[1]=(p.y-cm.intrinsic.ptr<double>(1)[2])*abSy;
			xsp[2]=1;
			
			for(unsigned int row=0; row<3; ++row)
			{
				xsp[row]-=cm.tvec.at<double>(row, 0);
			}
			
			for(unsigned int row=0; row<3; ++row)
			{
				
			   xp[row]+=cm.rotT.at<double>(row, 0)*xsp[0];
			   xp[row]+=cm.rotT.at<double>(row, 1)*xsp[1];
			   xp[row]+=cm.rotT.at<double>(row, 2)*xsp[2];
			   //xp[row]-=tvec.at<double>(row, 0);
			}
			
			//cout << xp[0] << " , " << xp[1] << " , " << xp[2] << endl;
			return Point3f(xp[0],xp[1],xp[2]);
			
			double lv[3]={0};
			lv[0]=xp[0]-cm.tvec.at<double>(0, 0);
			lv[1]=xp[1]-cm.tvec.at<double>(1, 0);
			lv[2]=xp[2]-cm.tvec.at<double>(2, 0);
			double lvs = sqrt( lv[0]*lv[0] + lv[1]*lv[1] + lv[2]*lv[2] );
			double nlv[3]={0};
			nlv[0]=lv[0]/lvs;
			nlv[1]=lv[1]/lvs;
			nlv[2]=lv[2]/lvs;
			
}

#define SCALE 10

static void draw_triangle( Mat& img , vector<Point2f> pls,Scalar color=Scalar( 255, 255, 255 ) ){
  int lineType = 8;

  /** Create some points */
  Point rook_points[1][pls.size()];
  for(int q=0;q<pls.size();q++){
	rook_points[0][q] = Point( pls[q].x, pls[q].y );
  }
  const Point* ppt[1] = { rook_points[0] };
  int npt[] = { pls.size() };

  fillPoly( img,
            ppt,
            npt,
            1,
            color,
            lineType );
            
}

static void cvt2bw(Mat &quadgoal,int thrs){
		cvtColor( quadgoal,quadgoal, CV_BGR2GRAY );
		threshold( quadgoal,quadgoal, thrs, 255,0 );
}

static void cm_plot_tdp(Mat &m,Cammat cm,Point2f wdp,Scalar color= Scalar(0,255,0)){
	
	float scale=SCALE;
	Point2f cp(m.cols/2,m.rows/2);
	
	Point3f tdp=cm_223(cm,wdp)*scale;
	
	vector<float> cpos(3);
	for(int q=0;q<3;q++){
			cpos[q]=0;
			for(int w=0;w<3;w++){
				cpos[q]+= cm.rotT.at<double>(q,w)*cm.tvec.at<double>(w, 0);
			}
			cpos[q]*=-1;
			cpos[q]*=scale;
	}
		
	Point3f ttp(cpos[0],cpos[1],cpos[2]);
	Point3f stdp = (tdp-ttp)*scale*2+ttp;
	
	Point2f tp(stdp.x,stdp.z*-1);
	Point2f sp(cpos[0],cpos[2]*-1);
	
	circle(m,cp+sp,3,Scalar(0,0,255));
	line(m,cp+sp,cp+tp, color , 1 ,1);	
	circle(m,cp,3,Scalar(255,0,255));
}

static void cm_plot_its_obeam(Mat &m,Cammat cm,OBeam ob1,OBeam ob2,int camno,Scalar color= Scalar(0,255,0)){
	Mat stc1 = Mat::zeros(m.rows,m.cols, CV_8UC3);
	Mat stc2 = Mat::zeros(m.rows,m.cols, CV_8UC3);
	float scale=SCALE;
	Point2f cp(m.cols/2,m.rows/2);
	
	
	for(int q=0;q<ob2.rays.size();q++){
		int w = (q+1) % ob2.rays.size();
		
		Point3f stdp1 = ob2.rays[q];
		Point3f stdp2 = ob2.rays[w];
		
		Point2f tp1(stdp1.x*scale,stdp1.z*scale*-1);
		Point2f tp2(stdp2.x*scale,stdp2.z*scale*-1);
		Point2f sp(ob2.pos.x*scale,ob2.pos.z*scale*-1);
		
		vector<Point2f> vpts;
		vpts.push_back(cp+sp);
		vpts.push_back(cp+tp1);
		vpts.push_back(cp+tp2);
		
		draw_triangle(stc2,vpts);
	}
	
	for(int q=0;q<ob1.rays.size();q++){
		int w = (q+1) % ob1.rays.size();
		
		Point3f stdp1 = ob1.rays[q];
		Point3f stdp2 = ob1.rays[w];
		
		Point2f tp1(stdp1.x*scale,stdp1.z*scale*-1);
		Point2f tp2(stdp2.x*scale,stdp2.z*scale*-1);
		Point2f sp(ob1.pos.x*scale,ob1.pos.z*scale*-1);
		
		vector<Point2f> vpts;
		vpts.push_back(cp+sp);
		vpts.push_back(cp+tp1);
		vpts.push_back(cp+tp2);
		
		draw_triangle(stc1,vpts);
	}
	
	imshow( str_concat("stc1 ", camno) , stc1);
	imshow( str_concat("stc2 ", camno) , stc2);
	
	cvt2bw(stc1,150);
	cvt2bw(stc2,150);
	Mat stc0=stc1.clone();
	bitwise_and(stc1,stc2,stc0);
	cvtColor( stc0,stc0, CV_GRAY2BGR );
	m+=stc0;
	
}
static void cm_plot_obeam(Mat &m,Cammat cm,OBeam ob,Scalar color= Scalar(0,255,0)){
	
	float scale=SCALE;
	Point2f cp(m.cols/2,m.rows/2);
	
	//Point3f tdp=cm_223(cm,wdp)*scale;
	
	vector<float> cpos(3);
	cpos[0]=ob.pos.x;
	cpos[1]=ob.pos.y;
	cpos[2]=ob.pos.z;
	for(int q=0;q<3;q++){
			cpos[q]*=scale;
	}
		
	Point3f ttp(cpos[0],cpos[1],cpos[2]);
	//Point3f stdp = (tdp-ttp)*scale*2+ttp;
	
	for(int q=0;q<ob.rays.size();q++){
		Point3f stdp = ob.rays[q];
		Point2f tp(stdp.x*scale,stdp.z*scale*-1);
		Point2f sp(cpos[0],cpos[2]*-1);
		
		circle(m,cp+sp,3,Scalar(0,0,255));
		line(m,cp+sp,cp+tp, color , 1 ,1);	
		circle(m,cp,3,Scalar(255,0,255));
	}
	
}

void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
    std::vector<cv::Point2f> top, bot;

    for (int i = 0; i < corners.size(); i++)
    {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }

    cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
    cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
    cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
    cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

    corners.clear();
    corners.push_back(tl);
    corners.push_back(tr);
    corners.push_back(br);
    corners.push_back(bl);
}

void do_sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
	vector<int> s;
	for(int j=0;j<corners.size();j++){
		s.push_back(j);
	}
	do {
        //std::cout << s << '\n';
        float suman=0;
        float crs=0;
        for(int jj=0;jj<corners.size();jj++){
			int j = s[jj];
			int k=s[(jj+1)%corners.size()];
			int l=s[(jj+2)%corners.size()];
				  
			Point vl1(corners[k].x-corners[j].x,corners[k].y-corners[j].y);
			Point vl2(corners[l].x-corners[k].x,corners[l].y-corners[k].y);
			
			crs=cross2d(vl1,vl2);
					
			float ang=acos( ( dot(normal(vl1),normal(vl2)) )/(1000.0*1000.0))/(2*3.14159)*360;
			suman+=ang;
        }
        if( abs( suman - 360 ) < 1 && crs>0){
			//cout << " -- " << suman << endl;
			std::vector<cv::Point2f> ncorners;
			for(int jj=0;jj<corners.size();jj++){
				int j = s[jj];
				ncorners.push_back( corners[j] );
			}
			corners = ncorners;
			
			 break;
		 }
    } while(std::next_permutation(s.begin(), s.end()));
    
}

bool myfunction (Vec4i i,Vec4i j) { return (distance(i[0],i[1],i[2],i[3])<distance(j[0],j[1],j[2],j[3])); }

static int do_validate_corners(vector<Point2f> corlist,Mat oimg1,int camno){
		Mat quad = Mat::zeros(400, 400, CV_8UC3);
		
		Mat quadgoal = Mat::zeros(400, 400, CV_8UC3);
		for(int q=0;q<100+300;q++){
				for(int w=0;w<100+300;w++){
					quadgoal.at<cv::Vec3b>(w,q)[0]=quadgoal.at<cv::Vec3b>(w,q)[1]=quadgoal.at<cv::Vec3b>(w,q)[2]=255;
				}
		}
		
		for(int q=100;q<100+200;q++){
				for(int w=100;w<100+200;w++){
					quadgoal.at<cv::Vec3b>(w,q)[0]=quadgoal.at<cv::Vec3b>(w,q)[1]=quadgoal.at<cv::Vec3b>(w,q)[2]=0;
				}
		}
		
		for(int q=100;q<100+100;q++){
				for(int w=100;w<100+100;w++){
					quadgoal.at<cv::Vec3b>(w,q)[0]=quadgoal.at<cv::Vec3b>(w,q)[1]=quadgoal.at<cv::Vec3b>(w,q)[2]=255;
				}
		}
		
		vector<Point2f> quad_pts;
		quad_pts.push_back(Point2f(0, 0));
		quad_pts.push_back(Point2f(quad.cols, 0));
		quad_pts.push_back(Point2f(quad.cols, quad.rows));
		quad_pts.push_back(Point2f(0, quad.rows));
		
		Mat transmtx = getPerspectiveTransform(corlist, quad_pts);
		warpPerspective(oimg1, quad, transmtx, quad.size());
		
		cvtColor( quad,quad, CV_BGR2GRAY );
		threshold( quad,quad, 160, 255,0 );
		
		cvtColor( quadgoal,quadgoal, CV_BGR2GRAY );
		threshold( quadgoal,quadgoal, 160, 255,0 );



		int ok=0;
		int times=0;
		for(int e=0;e<4;e++){
			int reds=0;
			
			for(int q=0;q<400;q++){
				for(int w=0;w<400;w++){
					if( abs(quad.at<uchar>(q,w) - quadgoal.at<uchar>(q,w) ) < 10 ){
						reds++;
					}
				}
			}
			
			if(reds>400*400*90/100){ok=1; break;}
			transpose(quad, quad);  
			flip(quad, quad,0);
			times++;
		}
		
		
		if(times>3){
			imshow( str_concat("quad - error ", camno) , quad);
			return -1;
		}
		else{
			imshow( str_concat("quad - ", camno) , quad);
			return times;
		}

}



static void do_get_cammat(Mat& plot,vector<Point2f> listc,int times,Mat mimg,Camera cam,Cammat& cm,vector<Point2f> &frust_plot,vector<Point2f>  &mark_plot,int camno){
		Mat quad = Mat::zeros(400, 400, CV_8UC3);
		
		
		vector<Point2f> quad_pts;
		quad_pts.push_back(Point2f(0, 0));
		quad_pts.push_back(Point2f(quad.cols, 0));
		quad_pts.push_back(Point2f(quad.cols, quad.rows));
		quad_pts.push_back(Point2f(0, quad.rows));
		
		Mat transmtx = getPerspectiveTransform(listc, quad_pts);

		int qidx[4];
		for(unsigned int q=0;q<listc.size();q++){
			float px=listc[q].x* transmtx.at<double>(0,0)+listc[q].y* transmtx.at<double>(0,1)+transmtx.at<double>(0,2);
			float py=listc[q].x* transmtx.at<double>(1,0)+listc[q].y* transmtx.at<double>(1,1)+transmtx.at<double>(1,2);
			float pt=listc[q].x* transmtx.at<double>(2,0)+listc[q].y* transmtx.at<double>(2,1)+transmtx.at<double>(2,2);
			px/=pt;
			py/=pt;
			
			px=(px<200)?0:1;
			py=(py<200)?0:1; 
			qidx[(int)py*2+(int)px]=q;
			
			//cout << px << " " << py << endl;
		}

		float an=3.14159/2.0*times;
		float det=transmtx.at<double>(0,0)*transmtx.at<double>(1,1)-transmtx.at<double>(0,1)*transmtx.at<double>(1,0);
		float a=transmtx.at<double>(1,1)/det;
		float b=-transmtx.at<double>(0,1)/det;
		float c=-transmtx.at<double>(1,0)/det;
		float d=transmtx.at<double>(0,0)/det;
		
		vector<Point2f> mark2d;
		vector<Point2f> imag2d;
		vector<Point3f> mark3d;
		
		
		mark2d.push_back(Point2f(0.0f,0.0f));
		mark3d.push_back(Point3f(2.0f,0.0f,2.0f));
		
		mark2d.push_back(Point2f(400.0f,0.0f));
		mark3d.push_back(Point3f(2.0f,0.0f,-2.0f));
		
		mark2d.push_back(Point2f(400.0f,400.0f));
		mark3d.push_back(Point3f(-2.0f,0.0f,-2.0f));
		
		mark2d.push_back(Point2f(0.0f,400.0f));
		mark3d.push_back(Point3f(-2.0f,0.0f,2.0f));
		
		
		mark2d.push_back(Point2f(100.0f,100.0f));
		mark3d.push_back(Point3f(1.0f,0.0f,1.0f));
		
		mark2d.push_back(Point2f(200.0f,100.0f));
		mark3d.push_back(Point3f(1.0f,0.0f,0.0f));
		
		mark2d.push_back(Point2f(300.0f,100.0f));
		mark3d.push_back(Point3f(1.0f,0.0f,-1.0f));
		
		mark2d.push_back(Point2f(100.0f,200.0f));
		mark3d.push_back(Point3f(0.0f,0.0f,1.0f));
		
		mark2d.push_back(Point2f(200.0f,200.0f));
		mark3d.push_back(Point3f(0.0f,0.0f,0.0f));
		
		mark2d.push_back(Point2f(300.0f,200.0f));
		mark3d.push_back(Point3f(0.0f,0.0f,-1.0f));
		
		mark2d.push_back(Point2f(100.0f,300.0f));
		mark3d.push_back(Point3f(-1.0f,0.0f,1.0f));
		
		mark2d.push_back(Point2f(200.0f,300.0f));
		mark3d.push_back(Point3f(-1.0f,0.0f,0.0f));
		
		mark2d.push_back(Point2f(300.0f,300.0f));
		mark3d.push_back(Point3f(-1.0f,0.0f,-1.0f));
		
		
		vector<pair<Point3f,Point3f>> cube3d;
		cube3d.push_back(make_pair( Point3f(-1.0f,0.0f,-1.0f) , Point3f(-1.0f,0.0f,1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(1.0f,0.0f,1.0f) , Point3f(-1.0f,0.0f,1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(-1.0f,0.0f,-1.0f) , Point3f(1.0f,0.0f,-1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(1.0f,0.0f,1.0f) , Point3f(1.0f,0.0f,-1.0f) ));
		
		
		
		
		cube3d.push_back(make_pair( Point3f(-1.0f,1.0f,-1.0f) , Point3f(-1.0f,1.0f,1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(1.0f,1.0f,1.0f) , Point3f(-1.0f,1.0f,1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(-1.0f,1.0f,-1.0f) , Point3f(1.0f,1.0f,-1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(1.0f,1.0f,1.0f) , Point3f(1.0f,1.0f,-1.0f) ));
		
		
		
		
		cube3d.push_back(make_pair( Point3f(-1.0f,0.0f,-1.0f) , Point3f(-1.0f,1.0f,-1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(1.0f,0.0f,1.0f) , Point3f(1.0f,1.0f,1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(-1.0f,0.0f,1.0f) , Point3f(-1.0f,1.0f,1.0f) ));
		
		
		cube3d.push_back(make_pair( Point3f(1.0f,0.0f,-1.0f) , Point3f(1.0f,1.0f,-1.0f) ));
		
		for(unsigned int q=0;q<listc.size();q++){
			int qw=(q+times)%listc.size();
			
			//imag2d.push_back( listc[qw] );
			//circle(quad,Point(wnpx,wnpy),3,Scalar(255,0,0 ));
			char numstr[512]; 
			sprintf(numstr,"%d (%d,%d) %d",q,(int)mark3d[qw].x,(int)mark3d[qw].z,times);
			putText(mimg,numstr ,listc[qw], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
			if(q==0)
			circle(mimg,listc[qw],5,Scalar(0,255,0 ));
			else
			circle(mimg,listc[qw],5,Scalar(255,0,0  ));
		}
		
		for(unsigned int q=0;q<mark2d.size();q++){
			float tx=transmtx.at<double>(0,2);
			float ty=transmtx.at<double>(1,2);
			float px=mark2d[q].x;
			float py=mark2d[q].y;
			
			if(px<1)
				px=0;	
			if(py<1)
				py=0;
			px-=200;
			py-=200;
			float npx = px*cos(an) + py*-1*sin(an);
			float npy = px*sin(an) + py*cos(an);
			npx+=200;
			npy+=200;
			px+=200;
			py+=200;
			
			//npx=(npx<200)?0:400;
			//npy=(npy<200)?0:400; 
			
			int tnpx=(npx<200)?0:1;
			int tnpy=(npy<200)?0:1; 
			//int idx=qidx[(int)tnpy*2+(int)tnpx];
			
			
			
			
			float wnpx;
			float wnpy;
			float otx;
			float oty;
			
			//float ub=100;
			//float lb=0;
			float s=1;
			//int sdir=0;
			
		
		
		for(int r=0;r<2000;r++){
			float tmnpx=npx;
			float tmnpy=npy;
			tmnpx*=s;
			tmnpy*=s;//s=1
			tmnpx-=tx;
			tmnpy-=ty;
			otx=a*tmnpx+b*tmnpy;
			oty=c*tmnpx+d*tmnpy;
			
			wnpx=otx* transmtx.at<double>(0,0)+oty* transmtx.at<double>(0,1)+tx;
			wnpy=otx* transmtx.at<double>(1,0)+oty* transmtx.at<double>(1,1)+ty;
			float wns=otx* transmtx.at<double>(2,0)+oty* transmtx.at<double>(2,1)+transmtx.at<double>(2,2);
			wnpx/=wns;
			wnpy/=wns;
			
			float errorx=(npx-wnpx);
			float errory=(npy-wnpy);
			//cout<<"==="<<endl;
			//cout<<errorx<<" "<<errory<<endl;
			//cout<<s<<" "<<lb<<" " <<ub<<endl;
			
			float ss=s;
			
			if(abs(errorx)>1){
				
				s+=0.01*( (errorx<0)?-1:1 );
				//if(errorx<0){//s มากไป
				//	ub=s;
				//}
				//else{//s น้อยไป
				//	lb=s;
				//}
				//s=(lb+ub)/2;
			}
			
			if(abs(errory)>1){
				
				s+=0.01*( (errory<0)?-1:1 );
				
				//if(errory<0){//s มากไป
				//	ub=s;
				//}
				//else{//s น้อยไป
				//	lb=s;
				//}
				//s=(lb+ub)/2;
			}
			if(ss==s){
				break;
			}
		}
		
			imag2d.push_back(Point2f(otx,oty));
			//circle(quad,Point(wnpx,wnpy),3,Scalar(255,0,0 ));
			char numstr[512]; 
			sprintf(numstr,"%d (%d,%d)",q,(int)mark3d[q].x,(int)mark3d[q].z);
			putText(mimg,numstr , Point(otx, oty), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
			circle(mimg,Point(otx,oty),3,Scalar(255,0,0 ));
		}
		
		Mat intrinsic = Mat(3, 3, CV_64FC1);
		Mat distCoeffs;
		Mat rvec, tvec;
		
		intrinsic.ptr<double>(0)[0] = 1;
		intrinsic.ptr<double>(1)[1] = 1;
		
		
		intrinsic.ptr<double>(0)[1] = 0;
		intrinsic.ptr<double>(0)[2] = (mimg.cols-1)*0.5;
		intrinsic.ptr<double>(1)[0] = 0;
		
		intrinsic.ptr<double>(1)[2] = (mimg.rows-1)*0.5;
		
		intrinsic.ptr<double>(2)[0] = 0;
		intrinsic.ptr<double>(2)[1] = 0;
		intrinsic.ptr<double>(2)[2] = 1;
		
		
		intrinsic=cam.intrinsic;
		distCoeffs=cam.distCoeffs;
		
		solvePnP(mark3d,imag2d,intrinsic,distCoeffs,rvec,tvec);
		//tvec.at<double>(1, 0)*=-1;
		cv::Mat rotation, viewMatrix(4, 4, CV_64F);
		cv::Rodrigues(rvec, rotation);
		
		Mat rotT=rotation.clone();
		transpose(rotT,rotT);
		
		vector<Point> patf;
		
		cm.rot = rotation.clone();
		cm.rotT = rotT.clone();
		cm.tvec = tvec.clone();
		cm.intrinsic = intrinsic.clone();
		
		//tvec.at<double>(w, 1)*=-1;
		
		
		for(unsigned int q=0;q<mark3d.size();q++){
			double p[3]={0};
			for(unsigned int row=0; row<3; ++row)
			{
				
			   p[row]+=rotation.at<double>(row, 0)*mark3d[q].x;
			   p[row]+=rotation.at<double>(row, 1)*mark3d[q].y;
			   p[row]+=rotation.at<double>(row, 2)*mark3d[q].z;
			   p[row]+=tvec.at<double>(row, 0);
			}
			
			p[0]/=p[2];
			p[1]/=p[2];
			
			double sp[2]={0};
			for(unsigned int row=0; row<2; ++row)
			{
			   sp[row]+=intrinsic.at<double>(row, 0)*p[0];
			   sp[row]+=intrinsic.at<double>(row, 1)*p[1];
			   sp[row]+=intrinsic.at<double>(row, 2);
			   //cout << intrinsic.at<double>(row, 2) << endl;
			}
			
			//cout << sp[0] << " " << sp[1] << endl;
			circle(mimg,Point(sp[0],sp[1]),3,Scalar(0,0,255 ));
			
			
			double px=sp[0],py=sp[1];//,pz=1;
			px-=intrinsic.ptr<double>(0)[2];
			py-=intrinsic.ptr<double>(1)[2];
			
		}
		
		vector<Point2f> aabb;
		aabb.push_back( Point2f( 0,0 ) );
		aabb.push_back( Point2f( mimg.cols-1 , 0 ) );
		aabb.push_back( Point2f( 0 , mimg.rows-1 ) );
		aabb.push_back( Point2f( mimg.cols-1 , mimg.rows-1 ) );
		
		for(int i=0;i<mimg.rows;i+=50){
			for(int j=0;j<mimg.cols;j+=50){
				aabb.push_back( Point2f( j , i ) );
			}
		}
		
		vector<Point3f> viewvec;
		//Mat plot = Mat::zeros(640,480, CV_8UC3);
		
		//cm_plot_tdp(plot,cm, cm_322( cm , Point3f( 0,0,0 ) )  );
		
		
		
		//cm_plot_tdp(plot,cm,Point2f( 0,0 ) );
		//cm_plot_tdp(plot,cm,Point2f(  mimg.cols-1 , 0 ) );
		//cm_plot_tdp(plot,cm,Point2f(  0 , mimg.rows-1 ) );
		for(unsigned int q=0;q<aabb.size();q++){
			
			
			//cm_plot_tdp(plot,cm,aabb[q]);
			frust_plot.push_back( aabb[q] );
			
			Point3f ttp(tvec.at<double>(0, 0),tvec.at<double>(1, 0),tvec.at<double>(2, 0));
			Point3f tdp = cm_223(cm,aabb[q]);
			//(tdp-ttp)*2+ttp
			Point2f wdp = cm_322(cm,tdp);
			circle(mimg,wdp,5,Scalar((int)(255*0),(int)(255*0),(int)(255*1) ));
			//circle(mimg,Point(sp1[0],sp1[1]),5,Scalar(0,255,255 ));
			//cout << wdp.x << " , " << wdp.y  << "         " << endl;
			
			
		}
		
		
		for(unsigned int q=0;q<cube3d.size();q++){
			Point3f mark3dp1=cube3d[q].first;
			Point3f mark3dp2=cube3d[q].second;
			
			//cm_plot_tdp(plot,cm, cm_322( cm , mark3dp1 ) , Scalar(255,0,0) );
			//cm_plot_tdp(plot,cm, cm_322( cm , mark3dp2 ) , Scalar(255,0,0) );
			
			double sp1[2]={0};
			double sp2[2]={0};
			
				{
					double p[3]={0};
					for(unsigned int row=0; row<3; ++row)
					{
						
					   p[row]+=rotation.at<double>(row, 0)*mark3dp1.x;
					   p[row]+=rotation.at<double>(row, 1)*mark3dp1.y*-1;
					   p[row]+=rotation.at<double>(row, 2)*mark3dp1.z;
					   p[row]+=tvec.at<double>(row, 0);
					}
					
					p[0]/=p[2];
					p[1]/=p[2];
					
					for(unsigned int row=0; row<2; ++row)
					{
					   sp1[row]+=intrinsic.at<double>(row, 0)*p[0];
					   sp1[row]+=intrinsic.at<double>(row, 1)*p[1];
					   sp1[row]+=intrinsic.at<double>(row, 2);
					}
					
					circle(mimg,Point(sp1[0],sp1[1]),3,Scalar(0,255,0 ));
				}
				
				{
					double p[3]={0};
					for(unsigned int row=0; row<3; ++row)
					{
						
					   p[row]+=rotation.at<double>(row, 0)*mark3dp2.x;
					   p[row]+=rotation.at<double>(row, 1)*mark3dp2.y*-1;
					   p[row]+=rotation.at<double>(row, 2)*mark3dp2.z;
					   p[row]+=tvec.at<double>(row, 0);
					}
					
					p[0]/=p[2];
					p[1]/=p[2];
					
					for(unsigned int row=0; row<2; ++row)
					{
					   sp2[row]+=intrinsic.at<double>(row, 0)*p[0];
					   sp2[row]+=intrinsic.at<double>(row, 1)*p[1];
					   sp2[row]+=intrinsic.at<double>(row, 2);
					}
					
					circle(mimg,Point(sp2[0],sp2[1]),3,Scalar(0,255,0 ));
				}
				
				line(mimg,Point(sp1[0],sp1[1]),Point(sp2[0],sp2[1]), Scalar(0,255,0) , 4 ,8);
		
				
		}
		
		//cm_plot_tdp(plot,cm, cm_322( cm , Point3f( 1,0,1 )) , Scalar(255,255,255)  );
		//cm_plot_tdp(plot,cm, cm_322( cm , Point3f( -1,0,1 )) , Scalar(255,255,255)  );
		//cm_plot_tdp(plot,cm, cm_322( cm , Point3f( 1,0,-1 )) , Scalar(255,255,255)  );
		//cm_plot_tdp(plot,cm, cm_322( cm , Point3f( -1,0,-1 )) , Scalar(255,255,255)  );
		
		mark_plot.push_back( cm_322( cm , Point3f( 0,0,0 ) ) );
		mark_plot.push_back(  cm_322( cm , Point3f( 1,0,1 )) );
		mark_plot.push_back(  cm_322( cm , Point3f( -1,0,1 )) );
		mark_plot.push_back(  cm_322( cm , Point3f( 1,0,-1 )) );
		mark_plot.push_back(  cm_322( cm , Point3f( -1,0,-1 )) );
		
		
		vector<float> cpos(3);
		for(int q=0;q<3;q++){
			cpos[q]=0;
			for(int w=0;w<3;w++){
				cpos[q]+= rotT.at<double>(q,w)*tvec.at<double>(w, 0);
			}
			cpos[q]*=-1;
			cpos[q]*=1.65;//ratio to real world
		}
		
		for(int q=0;q<3;q++){
			cout << "campos " << q << " = " << cpos[q] << endl;
		}
		
		
		//createOpenGLMatrixFrom(&OpenGLMatrix, rotation,tvec);
		/*for(unsigned int row=0; row<3; ++row)
		{
		   for(unsigned int col=0; col<3; ++col)
		   {
			  cout << rotation.at<double>(row, col) << " ";
		   }
		   cout << endl;
		}*/
		
		
		imshow(str_concat("mimg - ",camno), mimg);

}

void do_kmean_corner(Mat &img1,vector<Point> qhulli,Point cp,vector<Point2f> &corlist,int kmean,int iter){

		vector< pair<int, int> > idxd;
		 vector< Point2f >  cc; // center of clusters
		 //vector< Point2f >  corlist; // center of clusters
		 for(int j=0;j<kmean;j++){
			 int idxj=j*qhulli.size()/kmean;
			cc.push_back( Point2f(qhulli[idxj].x,qhulli[idxj].y) );
		 }
		 
		 /*
		 for(int j=0;j<qhull[i].size();j++){//all point
			 float dist=distance(qhull[i][j].x,qhull[i][j].y,cp.x,cp.y);
			 
			  circle(img1,qhull[i][j],3,Scalar(0,0,dist/3));
			  
			 idxd.push_back(make_pair( j , dist ));
		 }*/
   for(int pp=0;pp<iter;pp++){
	   
	   vector< pair<int ,int >  > npk;
		for(int j=0;j<kmean;j++){
			npk.push_back( make_pair( j , 0 ) );
		 }
	   
		 for(unsigned int j=0;j<qhulli.size();j++){//all point
			 float mind=9999;
			 float tdis=0;
			 int ink=0;
			 for(int k=0;k<kmean;k++){
				float dist=distance(qhulli[j].x,qhulli[j].y,cc[k].x,cc[k].y);
				//dist=dist*dist;
				if(dist < mind){
					mind=dist;
					tdis=distance(qhulli[j].x,qhulli[j].y,cp.x,cp.y);
					ink=k;
				}
			 }
			 //npk[ink].second++;
			 npk[ink].second+=tdis;
			 idxd.push_back(make_pair( j , ink ));
		 }
		 
		 sort(npk.begin(), npk.end(),
     [](const pair<int, int>& lhs, const pair<int, int>& rhs) -> bool {
             if (lhs.second == 0)
                 return true;
             return lhs.second < rhs.second; } );
         
        
		 
		 Point ncp(0,0);
		 for(int k=0;k<kmean;k++){
			 float npk=0;
			 Point sump(0,0);
			 for(unsigned int j=0;j<idxd.size();j++){
				 if(idxd[j].second == k ){
					 npk++;
					 //int ink=k;
					 Scalar color;
					 
					 if(k==0)
						color = Scalar(255,0,0);
					 if(k==1)
						color = Scalar(0,255,0);
					 if(k==2)
						color = Scalar(0,0,255);
					 if(k==3)
						color = Scalar(255,0,255);
					 if(k==4)
						color = Scalar(255,255,0);
					 if(k==5)
						color = Scalar(0,255,255);
					 if(k==6)
						color = Scalar(255,255,255);
						
					 circle(img1, qhulli[ idxd[j].first ] ,3,color);
					 sump += qhulli[ idxd[j].first ];
				 }
			 }
			 cc[k] = Point2f(sump.x/npk,sump.y/npk);
			 ncp+=Point(cc[k].x,cc[k].y);
			 Scalar color = Scalar(255,255,0);
			 circle(img1, cc[k] ,2,color);
		 }
		 
		 
		  
         
         
         
		 
		 for(int k=0;k<kmean;k++){
			 float mind=9999;
			 int inj=0;
			 for(unsigned int j=0;j<qhulli.size();j++){//all point
				float dist=distance(qhulli[j].x,qhulli[j].y,cc[k].x,cc[k].y);
				//dist=dist*dist;
				if(dist < mind){
					mind=dist;
					inj=j;
				}
			 }
			 cc[k]=Point2f(qhulli[inj].x,qhulli[inj].y);
		 }
		 
		 corlist.clear();
		 //for(int q=0;q<10;q++){
			 //Point ncp(0,0);
		//cout << endl;
         for(int qj=0;qj<4;qj++){
			 int kk = npk.size()-qj-1;
			//cout << npk[kk].first << " " << npk[kk].second << endl;
			int k=npk[kk].first;
			 //for(int k=0;k<kmean;k++)
			 {
				 float maxd=0;
				 int midx=0;
				 for(unsigned int j=0;j<idxd.size();j++){
					 if(idxd[j].second == k ){
						 int idxj = idxd[j].first;
						float dist=distance(qhulli[ idxj ].x,qhulli[ idxj ].y,cp.x,cp.y);
						if(dist > maxd){
							maxd=dist;
							midx=idxj;
						}
						
						 Scalar color;
					    dist/=250;
						 if(qj==0)
							color = Scalar(255*dist,0,0);
						 if(qj==1)
							color = Scalar(0,255*dist,0);
						 if(qj==2)
							color = Scalar(0,0,255*dist);
						 if(qj==3)
							color = Scalar(255*dist,0,255*dist);
						 if(qj==4)
							color = Scalar(255*dist,255*dist,0);
						 if(qj==5)
							color = Scalar(0,255*dist,255*dist);
						 if(qj==6)
							color = Scalar(255*dist,255*dist,255*dist);
							
						 circle(img1, qhulli[ idxj ] ,3,color);
							
					 }
				 }
				 
				 Scalar color = Scalar(0,255,255);
				 //if(pp==iter-1)
				 circle(img1, qhulli[ midx ] ,2,color);
					
				 corlist.push_back( Point2f(qhulli[ midx ].x , qhulli[ midx ].y));
				 
				 circle(img1, cp ,2,color);
				 //ncp+=qhulli[ midx ];
			 }
	    }
			 //cp=Point(ncp.x/kmean,ncp.y/kmean);
		//}
		 
		 idxd.clear();
		 
		 //cout << corlist.size() << endl;
    }
    
   
    
}

static void pl_get_bk(Mat& plot){
	plot = Mat::zeros(640,480, CV_8UC3);
	Point2f cp(plot.cols/2,plot.rows/2);
	float scale=SCALE;
	
	for(int i=0;i<scale*2;i++)
		for(int j=0;j<scale*2;j++)
			plot.at<Vec3b>(i+cp.x,j+cp.y)[0]=plot.at<Vec3b>(i+cp.x,j+cp.y)[1]=plot.at<Vec3b>(i+cp.x,j+cp.y)[2]=255;
			
}

static void pl_draw_mark(Mat& plot){
	Point2f cp(plot.cols/2,plot.rows/2);
	float scale=SCALE;
	
	Point2f sp;
	Point2f dp;
	
	sp = Point2f(-scale,-scale);
	dp = Point2f(scale,-scale);
	line(plot,sp+cp,dp+cp, Scalar(50,50,50) , 1 ,1);
	sp = Point2f(scale,scale);
	dp = Point2f(scale,-scale);
	line(plot,sp+cp,dp+cp, Scalar(50,50,50) , 1 ,1);
	sp = Point2f(scale,scale);
	dp = Point2f(-scale,scale);
	line(plot,sp+cp,dp+cp, Scalar(50,50,50) , 1 ,1);
	sp = Point2f(-scale,-scale);
	dp = Point2f(-scale,scale);
	line(plot,sp+cp,dp+cp, Scalar(50,50,50) , 1 ,1);
	
	
	scale=SCALE*2;
	
	sp = Point2f(-scale,-scale);
	dp = Point2f(scale,-scale);
	line(plot,sp+cp,dp+cp, Scalar(200,200,200) , 1 ,1);
	sp = Point2f(scale,scale);
	dp = Point2f(scale,-scale);
	line(plot,sp+cp,dp+cp, Scalar(200,200,200) , 1 ,1);
	sp = Point2f(scale,scale);
	dp = Point2f(-scale,scale);
	line(plot,sp+cp,dp+cp, Scalar(200,200,200) , 1 ,1);
	sp = Point2f(-scale,-scale);
	dp = Point2f(-scale,scale);
	line(plot,sp+cp,dp+cp, Scalar(200,200,200) , 1 ,1);
}

int count_white(Mat gray){
	int count_black = 0;
	int count_white = 0;
	for( int y = 0; y < gray.rows; y++ ) {
	  for( int x = 0; x < gray.cols; x++ ) {
		//if ( mask.at<uchar>(y,x) != 0 ) {
		  if ( gray.at<uchar>(y,x) == 255 ) {
			count_white++;
		  } 
		  else {
			count_black++;
		  } 
	  }
	}
	return count_white;
}

int do_label(Mat &gray){
	int nlb=1;
	for( int y = 0; y < gray.rows; y++ ) {
	  for( int x = 0; x < gray.cols; x++ ) {
		  if ( gray.at<uchar>(y,x) == 255 ) {
			  int ow=count_white(gray);
			  floodFill(gray, Point(x,y), Scalar(nlb) );
			  int nw=count_white(gray);
			  if(ow!=nw){
				  nlb++;
			  }
		  }
		  else {
		  } 
	  }
	}
	return nlb-1;
}

void do_detect(Mat oimg1,Camera cam,Cammat& cm,vector< vector<Point2f> > &obj_plot,int camno=0){

			 vector<Mat> rgb;
			 split(oimg1, rgb);
			 
			Mat robj = rgb[2] - rgb[1] - rgb[0];
			threshold( robj,robj, 50, 255,0 );
			/*
			cv::Mat dist;
			cv::distanceTransform(robj, dist, CV_DIST_L2, 3);
			cv::normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
			
			cv::threshold(dist, dist, .5, 1., CV_THRESH_BINARY);

			// Create the CV_8U version of the distance image
			// It is needed for cv::findContours()
			cv::Mat dist_8u;
			dist.convertTo(dist_8u, CV_8U);

			// Find total markers
			std::vector<std::vector<cv::Point> > contours;
			cv::findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			
			
			// Total objects
			int ncomp = contours.size();
			
			cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);
			for (int i = 0; i < ncomp; i++)
				cv::drawContours(markers, contours, i, cv::Scalar::all(i+1), -1);
			
			
			cv::circle(markers, cv::Point(5,5), 3, CV_RGB(255,255,255), -1);
			cv::circle(markers, cv::Point(markers.cols-5,5), 3, CV_RGB(255,255,255), -1);
			cv::circle(markers, cv::Point(5,markers.rows-5), 3, CV_RGB(255,255,255), -1);
			cv::circle(markers, cv::Point(markers.cols-5,markers.rows-5), 3, CV_RGB(255,255,255), -1);
			
			
			cv::imshow(str_concat("mark - ",camno),markers );
			
			cv::watershed(oimg1, markers);
			
			
			// Generate random colors
			std::vector<cv::Vec3b> colors;
			for (int i = 0; i < ncomp; i++)
			{
				int b = cv::theRNG().uniform(0, 255);
				int g = cv::theRNG().uniform(0, 255);
				int r = cv::theRNG().uniform(0, 255);

				colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
			}

			// Create the result image
			cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
			
			*/
			
			vector<Vec4i> qhierarchy;
			vector<vector<Point> > qctrs;
			Mat qtmp1=robj.clone();
			//cvtColor( qtmp1,qtmp1, CV_BGR2GRAY );
			//threshold( qtmp1,qtmp1, 160, 255,0 );
			findContours( qtmp1, qctrs, qhierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			
			for(unsigned int q=0;q<qctrs.size();q++){
				vector<Point2f> obj_plotx;
				int nps=10;
				int step=qctrs[q].size()/nps;
				if(qctrs[q].size()<=nps){
					step=1;
				}
				
				for(unsigned int w=0;w<qctrs[q].size();w+=step){
					obj_plotx.push_back(Point2f( qctrs[q][w].x, qctrs[q][w].y ));
				}
				obj_plot.push_back(obj_plotx);
			}
			//return ;
				
				
				
			/*
			vector<vector<Point> > qhull( qctrs.size() );
			for( unsigned int i = 0; i < qctrs.size(); i++ )
			{  convexHull( Mat(qctrs[i]), qhull[i], false ); }
			*/
			
			
			Mat markers=robj.clone();
			
			
			for(unsigned int q=0;q<qctrs.size();q++){
				for(unsigned int w=0;w<qctrs[q].size();w++){
					int e=(w+1)%qctrs[q].size();
					line(markers,  qctrs[q][w], qctrs[q][e], cv::Scalar(255, 0, 0), 1, CV_AA);
				}
			}
			
				
			
			//connectedComponents(markers,markers,4);
			
			/*
			vector<Point> mn;
			vector<Point> mx;
			int ncomp=do_label(markers);
			//cout << ncomp << endl;
			for(int i=0;i<ncomp;i++){
				mn.push_back( Point(999,999) );
				mx.push_back( Point(0,0) );
			}
			
			
			
			// Fill labeled objects with random colors
			for (int i = 0; i < markers.rows; i++)
			{
				for (int j = 0; j < markers.cols; j++)
				{
					int index = markers.at<uchar>(i,j);
					if (index > 0 && index <= ncomp){
						//dst.at<cv::Vec3b>(i,j) = colors[index-1];
						markers.at<uchar>(i,j)=255;
						
						if(mn[index-1].x>j){
							mn[index-1].x=j;
						}
						if(mn[index-1].y>i){
							mn[index-1].y=i;
						}
						if(mx[index-1].x<j){
							mx[index-1].x=j;
						}
						if(mx[index-1].y<i){
							mx[index-1].y=i;
						}
					}
					else{
						//dst.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
						
					}
				}
			}
			
			for(int i=0;i< ncomp ; i++){
				cv::line(markers, Point( mn[i].x, mn[i].y ), Point(mx[i].x,mn[i].y), cv::Scalar(255, 0, 0), 1, CV_AA);
				cv::line(markers, Point( mx[i].x, mx[i].y ), Point(mx[i].x,mn[i].y), cv::Scalar(255, 0, 0), 1, CV_AA);
				cv::line(markers, Point( mx[i].x, mx[i].y ), Point(mn[i].x,mx[i].y), cv::Scalar(255, 0, 0), 1, CV_AA);
				cv::line(markers, Point( mn[i].x, mn[i].y ), Point(mn[i].x,mx[i].y), cv::Scalar(255, 0, 0), 1, CV_AA);
				
				vector<Point2f> obj_plotx;
				
				obj_plotx.push_back(Point2f( mn[i].x, mn[i].y ));
				obj_plotx.push_back(Point2f(mx[i].x,mn[i].y));
				obj_plotx.push_back(Point2f( mx[i].x, mx[i].y ));
				obj_plotx.push_back(Point2f(mn[i].x,mx[i].y));
				
				float ar= (mx[i].x-mn[i].x)*(mx[i].y-mn[i].y) ;
				if(ar>400){
					obj_plot.push_back(obj_plotx);
				}
			}*/
			
			
			

			cv::imshow(str_concat("markers - ",camno), markers);
			
			/*
			std::vector<cv::Point> points;
			cv::Mat_<uchar>::iterator it = robj.begin<uchar>();
			cv::Mat_<uchar>::iterator end = robj.end<uchar>();
			for (; it != end; ++it)
				if (*it)
					points.push_back(it.pos());
					
			cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
			cv::Point2f vertices[4];
			  box.points(vertices);
			  
			  vector< Point2f > bb;
			  
			  for(int i = 0; i < 4; ++i){
				cv::line(robj, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);
				bb.push_back( vertices[i] );
				//cm_plot_tdp(plot,cm,vertices[i],Scalar(0,0,255));
			  }
			  
			  if(bb.size()>0){
					//obj_plot.push_back(bb);
			  }
			  
			  */
			 imshow(str_concat("red obj - ",camno),  robj);
			 //cout << " end detect" << endl;
}

int selfcalib_dummy(Mat &plot,Mat img1,Camera cam,Cammat& cm,vector<Point2f> &frust_plot,vector<Point2f>  &mark_plot,vector< vector<Point2f> > &obj_plot,int camno=0,int rtrun=0){
	//Mat img1 = imread("cap_image/cam1_conf10.png", CV_LOAD_IMAGE_COLOR);
	
	
	Mat oimg1=img1.clone();
	Mat mimg=img1.clone();
	//pre process
	//cvtColor(img1, img1, CV_BGR2HSV);    


int mark_found=0;
	
vector<Vec4i> qhierarchy;
vector<vector<Point> > qctrs;

Mat qtmp1=img1.clone();
cvtColor( qtmp1,qtmp1, CV_BGR2GRAY );

threshold( qtmp1,qtmp1, 160, 255,0 );

for(int i=0;i<qtmp1.rows;i+=100)
	floodFill(qtmp1,Point(0,i),Scalar(0,0,0));
for(int j=0;j<qtmp1.cols;j+=100)
	floodFill(qtmp1,Point(j,0),Scalar(0,0,0));
for(int i=0;i<qtmp1.rows;i+=100)
	floodFill(qtmp1,Point(qtmp1.cols-1,i),Scalar(0,0,0));
for(int j=0;j<qtmp1.cols;j+=100)
	floodFill(qtmp1,Point(j,qtmp1.rows-1),Scalar(0,0,0));

imshow( str_concat("qtmp1 - ",camno) , qtmp1);

//return 1;
		
Mat qtmp2=qtmp1.clone();




//findContours( qtmp1, qctrs, qhierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
findContours( qtmp1, qctrs, qhierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


img1=Mat::zeros(img1.rows,img1.cols,CV_8UC3);

/// Find the convex hull object for each contour
   vector<vector<Point> > qhull( qctrs.size() );
	#pragma omp parallel for		
   for( unsigned int i = 0; i < qctrs.size(); i++ )
      {  convexHull( Mat(qctrs[i]), qhull[i], false ); }
      #pragma omp barrier
vector<vector<Point> > aqhull;



      //cout << " asdfdas" << endl;
for( unsigned int r = 0; r < qhull.size(); r++ )
      {
		  int i=(r+5)% qhull.size();
		  
		  Point cp(qhull[i][0].x,qhull[i][0].y);
		  
			for(unsigned int j=1;j<qhull[i].size();j++){
				cp+=qhull[i][j];
			}
			cp.x/=qhull[i].size();
			cp.y/=qhull[i].size();
			float area=0;
			for(unsigned int j=0;j<qhull[i].size()-1;j++){
				Point u=qhull[i][j]-cp;
				Point v=qhull[i][j+1]-cp;
				area+= abs(u.x*v.y-u.y*v.x)/2;
			}
			
		//float w=1280;
		//float h=720;
			
		  //float inan=0;
		  //int ncor=0;
		  //float len=0;
		  
		  //float sumlen=0;
		  
		  vector<float> vf;
		  //int corn=0;
		  
		  //int nearcn=0;
		  
		  Point lp;
		  //int init=1;
		  
		  vector<Point2f> listc;
		  
		  vector<Point2f> listcd;
		 
		
		//Less Area
		if(area < 50*50){
			continue;
		}
		
		 //init
		 
		 vector< Point2f >  corlist; // corners list
		
		vector<Point> tqhi;
		int nql=qhull[i].size();
		for(int q=0;q<nql;q++){
			int w=(q+1)%nql;
			float dist=distance(qhull[i][q].x,qhull[i][q].y,qhull[i][w].x,qhull[i][w].y);
			float dist2=distance(qhull[i][q].x,qhull[i][q].y,cp.x,cp.y)/100;
			if(dist<10){
				for(int pp=0;pp<5+dist2*dist2;pp++){
					tqhi.push_back(Point( (qhull[i][q].x+qhull[i][w].x)/2 , (qhull[i][q].y+qhull[i][w].y)/2 ));
				}
			}
			else{
				int ndd=dist/10;
				for(int r=0;r<ndd;r++){
					tqhi.push_back(Point( (qhull[i][q].x-qhull[i][w].x)*r/ndd +qhull[i][w].x , (qhull[i][q].y-qhull[i][w].y)*r/ndd +qhull[i][w].y  ));
				}
			}
		}
		qhull[i]=tqhi;
		
		 cp = Point(qhull[i][0].x,qhull[i][0].y);
		  
			for(unsigned int j=1;j<qhull[i].size();j++){
				cp+=qhull[i][j];
			}
			cp.x/=qhull[i].size();
			cp.y/=qhull[i].size();
			
		do_kmean_corner(img1,qhull[i],cp,corlist,9,500);//4cluster 300iteration
		cout << "Before do_sortCorner " << endl;
		do_sortCorners(corlist, cp);
		//sortCorners(corlist, cp);
		for(unsigned int qr=0;qr<corlist.size();qr++){
			int r1=qr%corlist.size();
			int r2=(qr+1)%corlist.size();
			line(img1,corlist[r1],corlist[r2], Scalar(0,255,0) , 4 ,8);
		}
		
		int times=do_validate_corners(corlist,oimg1,camno);
		if(times > -1){
			//pass 
			for(unsigned int qr=0;qr<corlist.size();qr++){
				int r1=qr%corlist.size();
				int r2=(qr+1)%corlist.size();
				line(img1,corlist[r1],corlist[r2], Scalar(255,0,255) , 4 ,8);
			}
			
			
			 do_get_cammat(plot,corlist,times,mimg,cam,cm,frust_plot,mark_plot,camno);
			 //Mat plot = Mat::zeros(640,480, CV_8UC3);
			 //imshow(str_concat("plot - ",camno), plot);
			 
			 vector<Mat> rgb;
			 split(oimg1, rgb);
			 
			Mat robj = rgb[2] - rgb[1] - rgb[0];
			threshold( robj,robj, 100, 255,0 );
			
			/*std::vector<cv::Point> points;
			cv::Mat_<uchar>::iterator it = robj.begin<uchar>();
			cv::Mat_<uchar>::iterator end = robj.end<uchar>();
			for (; it != end; ++it)
				if (*it)
					points.push_back(it.pos());
					
			cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
			cv::Point2f vertices[4];
			  box.points(vertices);
			  
			  vector< Point2f > bb;
			  
			  for(int i = 0; i < 4; ++i){
				cv::line(robj, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);
				bb.push_back( vertices[i] );
				//cm_plot_tdp(plot,cm,vertices[i],Scalar(0,0,255));
			  }
			  
			  if(bb.size()>0){
					obj_plot.push_back(bb);
			  }
			  */
			 imshow(str_concat("red obj - ",camno),  robj);
			 //do_get_aabb_obj(oimg1);
			 mark_found=1;
		}
		
		
		
		imshow(str_concat("img1 - ",camno), img1);
        
		 
		 
		 continue;
		
	  }


if(mark_found==0){
	cout << " mark not found." << endl;
	return -1;
}
//cout << "% " << endl;
imshow(str_concat("img1 - ",camno), img1);
imshow(str_concat("oimg1 - ",camno), oimg1);
//cout << "^ " << endl;
return 1;

}

static void find_moving_obj(Mat imgB,Mat imgA){//this frame , bg model
	// Load two images and allocate other structures
		
		
		//Mat imgA = ovimg[0].clone();
		//Mat imgB = vimg[0].clone();
		
		
		/*
		static float gauss_sum[1281][721][3];
		static float gauss_sumsq[1281][721][3];
		static float gauss_sd[1281][721][3];
		static Mat gauss_mean=cv::Mat::zeros(imgA.rows, imgA.cols, CV_8UC3);
		static Mat gauss_diff=cv::Mat::zeros(imgA.rows, imgA.cols, CV_8UC3);
		static Mat gauss_matsd=cv::Mat::zeros(imgA.rows, imgA.cols, CV_8UC3);
		static float n=0;
		
		
		int upd=1;
		for(unsigned int py=0;py<imgB.rows;py++){
			for(unsigned int px=0;px<imgB.cols;px++){
					
					float diff=sqrt(
					(imgB.data[(py*imgB.cols+px)*3+0] - gauss_mean.data[(py*imgB.cols+px)*3+0])*
					(imgB.data[(py*imgB.cols+px)*3+0] - gauss_mean.data[(py*imgB.cols+px)*3+0])
					+
					(imgB.data[(py*imgB.cols+px)*3+1] - gauss_mean.data[(py*imgB.cols+px)*3+1])*
					(imgB.data[(py*imgB.cols+px)*3+1] - gauss_mean.data[(py*imgB.cols+px)*3+1])
					+
					(imgB.data[(py*imgB.cols+px)*3+2] - gauss_mean.data[(py*imgB.cols+px)*3+2])*
					(imgB.data[(py*imgB.cols+px)*3+2] - gauss_mean.data[(py*imgB.cols+px)*3+2])
					);
					
					float sd=sqrt( gauss_sd[px][py][0]* gauss_sd[px][py][0]
					+gauss_sd[px][py][1]* gauss_sd[px][py][1]
					+gauss_sd[px][py][2]* gauss_sd[px][py][2]);
					
					if(abs(diff) > 20*3 ){
						upd--;
						gauss_diff.data[(py*imgB.cols+px)*3+0]=imgB.data[(py*imgB.cols+px)*3+0];
						gauss_diff.data[(py*imgB.cols+px)*3+1]=imgB.data[(py*imgB.cols+px)*3+1];
						gauss_diff.data[(py*imgB.cols+px)*3+2]=imgB.data[(py*imgB.cols+px)*3+2];
					}
					else{
						//cout << "diff frame" << endl;
						gauss_diff.data[(py*imgB.cols+px)*3+0]=
						gauss_diff.data[(py*imgB.cols+px)*3+1]=
						gauss_diff.data[(py*imgB.cols+px)*3+2]=0;
					}
					
					gauss_matsd.data[(py*imgB.cols+px)*3+0]=
						gauss_matsd.data[(py*imgB.cols+px)*3+1]=
						gauss_matsd.data[(py*imgB.cols+px)*3+2]=abs(sd);
					
			}
		}
		
		static int init_st=1;
		if(n>=50&&init_st==1){
			init_st=0;
			cout << " init complete." << endl;
		}
		
		if(upd>-300||n<50){
			n++;
			for(unsigned int py=0;py<imgB.rows;py++){
				for(unsigned int px=0;px<imgB.cols;px++){
					for(unsigned int dep=0;dep<3;dep++){
						
						
						gauss_sum[px][py][dep] += imgB.data[(py*imgB.cols+px)*3+dep];
						gauss_mean.data[(py*imgB.cols+px)*3+dep] = gauss_sum[px][py][dep]/n;
						
						float dif=imgB.data[(py*imgB.cols+px)*3+dep]- gauss_mean.data[(py*imgB.cols+px)*3+dep];
						gauss_sumsq[px][py][dep] += dif*dif;
						gauss_sd[px][py][dep] = sqrt(  gauss_sumsq[px][py][dep]/n  );
						
						
						
					}
				}
			}
			
			
		}
		else{
			cout << "diff " << upd << endl;
			namedWindow( "gauss_diff", 0 );
			imshow( "gauss_diff", gauss_diff );
		}
		
		namedWindow( "gauss_matsd", 0 );
		imshow( "gauss_matsd", gauss_matsd );
		
		namedWindow( "gauss mean", 0 );
		imshow( "gauss mean", gauss_mean );
		*/
		
		
		static Mat ofrm=imgA.clone();//ovimg[0].clone();
		static int init_p=20;
		static Mat ofrm_edge=ofrm.clone();
		
		if(init_p>0){
			init_p--;
			if(init_p==0)
				cout << " start!" << endl;
			ofrm==imgA.clone();//ovimg[0].clone();
			ofrm_edge=ofrm.clone();
			//medianBlur(ofrm,ofrm,3);
			//blur( ofrm,ofrm, Size(3,3) );
			cvtColor(ofrm,ofrm, CV_BGR2HSV);
			
			cvtColor(ofrm_edge,ofrm_edge, CV_BGR2GRAY);
			Canny( ofrm_edge,ofrm_edge, 50, 50*3, 3 );
			blur( ofrm_edge,ofrm_edge, Size(5,5) );
			cvtColor(ofrm_edge,ofrm_edge, CV_GRAY2BGR);
		}
		
		Mat frm=imgB.clone();//vimg[0].clone();
		//medianBlur(frm,frm,3);
		//blur( frm,frm, Size(3,3) );
		cvtColor(frm,frm, CV_BGR2HSV);
		
		
		Mat fdiff=cv::Mat::zeros(frm.rows, frm.cols, CV_8UC3);
		
		float htg_frm[900]={0};
		float htg_ofrm[900]={0};
		
		float htg_conv[900]={0};
		
		float htg_frm_max=0;
		float htg_ofrm_max=0;
		
		for(int py=0;py<frm.rows;py++){
				for(int px=0;px<frm.cols;px++){
					htg_ofrm[ofrm.data[(py*ofrm.cols+px)*3+1]]++;
					htg_frm[frm.data[(py*frm.cols+px)*3+1]]++;
				}
		}
		
		//cout << htg_frm_max << " " << htg_ofrm_max << endl;
		for(int py=0;py<900;py++){
			if(htg_frm_max < htg_frm[py])
				htg_frm_max = htg_frm[py];
			
			if(htg_ofrm_max < htg_ofrm[py])
				htg_ofrm_max = htg_ofrm[py];
			//cout << htg_frm[py] << " " << htg_ofrm[py] << endl;
		}
		
		for(int py=0;py<900;py++){
			htg_frm[py]/=(float)htg_frm_max;
			htg_ofrm[py]/=(float)htg_ofrm_max;
			//cout << htg_frm[py] << " " << htg_ofrm[py] << endl;
		}
		
		int cwb=0;
		float cwb_max=0;
		
		
		for(int py=0;py<900;py++){//conv
				int pp=py-300;//back -100
				float rconv=0;
				for(unsigned int pz=0;pz<300;pz++){
					unsigned int pq=pp+pz;
					if(pq>=0&&pq<=255){
						rconv+=htg_frm[pq]*htg_ofrm[pz];
					}
				}
				htg_conv[py]=rconv;
				if(rconv>cwb_max){
					cwb_max=rconv;
					cwb=pp;
				}
		}
		
		//cout << "cwb " << cwb << endl;
		for(int py=0;py<frm.rows;py++){
				for(int px=0;px<frm.cols;px++){
					htg_ofrm[ofrm.data[(py*ofrm.cols+px)*3+1]]+=cwb;
				}
		}
		
		
		Mat sh_htg_frm=cv::Mat::zeros(20, 900*2, CV_8UC1);
		Mat sh_htg_ofrm=cv::Mat::zeros(20, 900*2, CV_8UC1);
		Mat sh_htg_conv=cv::Mat::zeros(20, 900*2, CV_8UC1);
			
		for(int pa=0;pa<900;pa++){
			for(int px=0;px<2;px++){
				for(int py=0;py<20;py++){
					sh_htg_frm.data[(py*sh_htg_frm.cols+(px+pa*2) )] = htg_frm[pa] * 255;
					sh_htg_ofrm.data[(py*sh_htg_ofrm.cols+(px+pa*2) )] = htg_ofrm[pa] * 255;
					sh_htg_conv.data[(py*sh_htg_frm.cols+(px+pa*2) )] = htg_conv[pa] * 10;
				}
			}
		}
		
		
		namedWindow( "sh_htg_frm", 0 );
		imshow( "sh_htg_frm", sh_htg_frm );
		
		namedWindow( "sh_htg_ofrm", 0 );
		imshow( "sh_htg_ofrm", sh_htg_ofrm );
		
		namedWindow( "sh_htg_conv", 0 );
		imshow( "sh_htg_conv", sh_htg_conv );
		
		
		namedWindow( "ofrm_edge", 0 );
		imshow( "ofrm_edge", ofrm_edge );
	
		
		for(unsigned int py=0;py<frm.rows;py++){
				for(unsigned int px=0;px<frm.cols;px++){
					float pi=3.14159625/180.0;
					float h2=frm.data[(py*frm.cols+px)*3+0]*pi;
					float h1=ofrm.data[(py*ofrm.cols+px)*3+0]*pi;
					
					float x1=ofrm.data[(py*ofrm.cols+px)*3+1]*cos(h1);
					float y1=ofrm.data[(py*ofrm.cols+px)*3+1]*sin(h1);
					float z1=ofrm.data[(py*ofrm.cols+px)*3+2];
					
					float x2=frm.data[(py*frm.cols+px)*3+1]*cos(h2);
					float y2=frm.data[(py*frm.cols+px)*3+1]*sin(h2);
					float z2=frm.data[(py*ofrm.cols+px)*3+2];
					
					float diff=sqrt(
					(x2-x1)*(x2-x1) +
					(y2-y1)*(y2-y1) +
					(z2-z1)*(z2-z1)
					);
					//4072
						fdiff.data[(py*frm.cols+px)*3+0]=diff/4;
						fdiff.data[(py*frm.cols+px)*3+1]=diff/4;
						fdiff.data[(py*frm.cols+px)*3+2]=diff/4;
					
				}
			}
		
		int erosion_size=2;
		erode( fdiff,fdiff, getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) ) );
			
		
		namedWindow( "fdiff", 0 );
		imshow( "fdiff", fdiff );
		
		
		cvtColor(imgA,imgA, CV_BGR2GRAY);
		cvtColor(imgB,imgB, CV_BGR2GRAY);
		
		Size img_sz = imgA.size();
		Mat imgC(img_sz,1);
		imgC=cv::Mat::zeros(imgA.rows, imgA.cols, CV_8UC3);
	 
		int win_size = 15;
		int maxCorners = 20; 
		double qualityLevel = 0.05; 
		double minDistance = 5.0; 
		int blockSize = 3; 
		double k = 0.04; 
		std::vector<cv::Point2f> cornersA; 
		cornersA.reserve(maxCorners); 
		std::vector<cv::Point2f> cornersB; 
		cornersB.reserve(maxCorners);
		
		//cout << "!" << endl;
		
		goodFeaturesToTrack( imgA,cornersA,maxCorners,qualityLevel,minDistance,cv::Mat());
		goodFeaturesToTrack( imgB,cornersB,maxCorners,qualityLevel,minDistance,cv::Mat());
	 
		//cout << "2" << endl;
		
		cornerSubPix( imgA, cornersA, Size( win_size, win_size ), Size( -1, -1 ), 
					  TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );
		
		cornerSubPix( imgB, cornersB, Size( win_size, win_size ), Size( -1, -1 ), 
					  TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );
	 
		// Call Lucas Kanade algorithm
		//cout << "3" << endl;
		
		CvSize pyr_sz = Size( img_sz.width+8, img_sz.height/3 );
	 
		std::vector<uchar> features_found; 
		features_found.reserve(maxCorners);
		std::vector<float> feature_errors; 
		feature_errors.reserve(maxCorners);
		
		//cout << "4" << endl;
		
		calcOpticalFlowPyrLK( imgA, imgB, cornersA, cornersB, features_found, feature_errors ,
			Size( win_size, win_size ), 5,
			 cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );
	 
		//cout << "5" << endl;
	 
		// Make an image of the results
	 
		for( int i=0; i < features_found.size(); i++ ){
				//cout<<"Error is "<<feature_errors[i]<<endl;
				//continue;
		
			//cout<<"Got it"<<endl;
			Point p0( ceil( cornersA[i].x ), ceil( cornersA[i].y ) );
			Point p1( ceil( cornersB[i].x ), ceil( cornersB[i].y ) );
			line( imgC, p0, p1, CV_RGB(255,255,255), 2 );
		}
	 
		namedWindow( "ImageA", 0 );
		namedWindow( "ImageB", 0 );
		namedWindow( "LKpyr_OpticalFlow", 0 );
	 
		imshow( "ImageA", imgA );
		imshow( "ImageB", imgB );
		imshow( "LKpyr_OpticalFlow", imgC );
	 
		
}



static void kalman_filter(Mat& in){
}

static void get_aabb_lists(Mat in){
}

static void selfcalib(vector<int> vcam){
	
	vector<VideoCapture> vcap;
	vector<int> vconf;
	vector<int> vcid;
	vector<Cammat> vcm;
	for(int i=0;i<vcam.size();i++){
		if(vcam[i]<=0) continue;
		VideoCapture capture = VideoCapture(i);
		capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
		vcap.push_back(capture);
		vconf.push_back(vcam[i]);
		vcid.push_back(i);
		Cammat cm;
		vcm.push_back(cm);
	}
	
	vector<Camera> vcf;
    for(int i=0;i<vconf.size();i++){
		Camera cam;
		cout << " Cam " << vcid[i] << " use conf no. " << vconf[i] << endl;
        // create and open an archive for input
        std::ifstream ifs(getConfigPath(CAMERA,vconf[i]));
        boost::archive::text_iarchive ia(ifs);
        // read class state from archive
        ia >> cam;
        // archive and stream closed when destructors are called
        vcf.push_back(cam);
    }
    
    //calibrateCamera(cam.object_points, cam.image_points, cam.image.size(), cam.intrinsic, cam.distCoeffs, cam.rvecs, cam.tvecs);
    Mat image;
    Mat imageUndistorted;
    vector<Mat> ovimg;
    
    //wait 50 frame - wait for stable frame

    //calibrate
		
	//each camera
		//moving object detect
			//kalman frame diff
			//listAABB - contour convexhull
		
		//convert listAABB to 4vector + 1 position
		//draw in y axis change color
    
    int unstable_frame=50;
    //start with 1 object
    int pass_multi_calib=0;
    
    //Wait for Stable state && Calibration 
    for(int q=0;q<unstable_frame||!pass_multi_calib;q++)
    {
		vector<Mat> vimg;
		for(int i=0;i<vcap.size();i++){
			vcap[i] >> image;
			undistort(image, imageUndistorted, vcf[i].intrinsic, vcf[i].distCoeffs);
			//
			vimg.push_back(imageUndistorted.clone());
			char numstr[512]; // enough to hold all numbers up to 64-bits
			sprintf(numstr,"cam%d - conf%d",vcid[i],vconf[i]);
			imshow(numstr, imageUndistorted);
		}
		static int s_init=1;
		if(s_init==1){
			s_init=0;
			ovimg=vimg;
		}
		
		int pass_count=0;
		if(q>unstable_frame){
			
			Mat plot;
			pl_get_bk(plot);
			
		vector<Point2f> frust_plot;
		vector<Point2f> mark_plot;
		vector< vector<Point2f> > obj_plot; //vector of "aabb"
			for(int i=0;i<vcap.size();i++){
				cout << "Cab " << i << endl;
				if(selfcalib_dummy(plot,vimg[i].clone(),vcf[i],vcm[i],frust_plot,mark_plot,obj_plot,i) > 0){
					pass_count++;
				}
				
			}
			pl_draw_mark(plot);
			imshow("plot - all", plot);
			
			
			if(pass_count==vcap.size()){
				 pass_multi_calib=1;
				 cout << " pass " << endl;
			}
		}
		//selfcalib_dummy(vimg[0].clone());
        
		//selfcalib_dummy(vimg[0].clone(),vcf[0]);
		//selfcalib_dummy(vimg[1].clone(),vcf[1]);
		ovimg=vimg;
		
    }
    
    
    //Detection
    while(1)
    {
		vector<Mat> vimg;
		for(unsigned int i=0;i<vcap.size();i++){
			vcap[i] >> image;
			undistort(image, imageUndistorted, vcf[i].intrinsic, vcf[i].distCoeffs);
			//
			vimg.push_back(imageUndistorted.clone());
			char numstr[512]; // enough to hold all numbers up to 64-bits
			sprintf(numstr,"cam%d - conf%d",vcid[i],vconf[i]);
			imshow(numstr, imageUndistorted);
		}
		static int s_init=1;
		if(s_init==1){
			s_init=0;
			ovimg=vimg;
		}
		
		
		//selfcalib_dummy(vimg[0].clone());
        
		//selfcalib_dummy(vimg[0].clone(),vcf[0]);
		//selfcalib_dummy(vimg[1].clone(),vcf[1]);
		Mat plot = Mat::zeros(640,480, CV_8UC3);
		
		vector < vector<Point2f> > frust_plots(vcap.size());
		vector < vector<Point2f> > mark_plots(vcap.size());
		vector < vector< vector<Point2f> > > obj_plots(vcap.size()); //vector of "aabb"
		vector < vector < OBeam > > obj_beams(vcap.size());
		for(unsigned int i=0;i<vcap.size();i++){
			//cout << " /?" << endl;
			    do_detect(vimg[i].clone(),vcf[i],vcm[i],obj_plots[i],i);
			    for(unsigned int j=0;j<obj_plots[i].size();j++){
					OBeam obm;
					obm.set_up(vcm[i],obj_plots[i][j]);
					obj_beams[i].push_back( obm );
				}
				//if(selfcalib_dummy(plot,vimg[i].clone(),vcf[i],vcm[i],frust_plots[i],mark_plots[i],obj_plots[i],i) > 0){
				//}
			//cout << " /??" << endl;
		}
		
		Mat plot_cofusion = Mat::zeros(30,30, CV_8UC3);
		for(unsigned int i=0;i<vcap.size();i++){
			for(unsigned int j=i+1;j<vcap.size();j++){
				
			    for(unsigned int k=0;k<obj_beams[i].size();k++){
					for(unsigned int q=0;q<obj_beams[j].size();q++){
						int ncld = obj_beams[i][k].collide(obj_beams[j][q]);
						plot_cofusion.at<Vec3b>(k,q)[0]=ncld;
						
						/*if(ncld>0){
							cm_plot_obeam(plot,vcm[i],obj_beams[i][k],Scalar(0,0,255));
							cm_plot_obeam(plot,vcm[i],obj_beams[j][q],Scalar(0,0,255));
						}
						else{*/
							cm_plot_obeam(plot,vcm[i],obj_beams[i][k],Scalar(0,255,0));
							cm_plot_obeam(plot,vcm[i],obj_beams[j][q],Scalar(0,255,0));
						//}
						cout << ncld << " ";
					}
					cout << endl;
				}
				
				//Match Max Collision
				for(unsigned int k=0;k<obj_beams[i].size();k++){
					int maxv=0;
					int maxq=-1;
					for(unsigned int q=0;q<obj_beams[j].size();q++){
						if( plot_cofusion.at<Vec3b>(k,q)[0] > maxv ){
								maxv = plot_cofusion.at<Vec3b>(k,q)[0];
								maxq = q;
						}
					}
					
					if(maxq>-1){
						cm_plot_obeam(plot,vcm[i],obj_beams[i][k],Scalar(0,0,255));
						cm_plot_obeam(plot,vcm[i],obj_beams[j][maxq],Scalar(0,0,255));
					}
				}
				
				
				for(unsigned int k=0;k<obj_beams[i].size();k++){
					int maxv=0;
					int maxq=-1;
					for(unsigned int q=0;q<obj_beams[j].size();q++){
						if( plot_cofusion.at<Vec3b>(k,q)[0] > maxv ){
								maxv = plot_cofusion.at<Vec3b>(k,q)[0];
								maxq = q;
						}
					}
					if(maxq>-1){
						cm_plot_its_obeam(plot,vcm[i],obj_beams[i][k],obj_beams[j][maxq],k*10+maxq,Scalar(0,0,255));
					}
				}
				
				for(unsigned int k=0;k<obj_beams[j].size();k++){
					int maxv=0;
					int maxq=-1;
					for(unsigned int q=0;q<obj_beams[i].size();q++){
						if( plot_cofusion.at<Vec3b>(q,k)[0] > maxv ){
								maxv = plot_cofusion.at<Vec3b>(q,k)[0];
								maxq = q;
						}
					}
					if(maxq>-1){
						cm_plot_its_obeam(plot,vcm[j],obj_beams[j][k],obj_beams[i][maxq],k*10+maxq,Scalar(0,0,255));
					}
				}
				
				for(unsigned int k=0;k<obj_beams[i].size();k++){
					for(unsigned int q=0;q<obj_beams[j].size();q++){
						plot_cofusion.at<Vec3b>(k,q)[0]*=10;
					}
				}
				//cout << "vcap   " << i << " , " << j << endl;
				
			}
		}
		
		resize(plot_cofusion,plot_cofusion,Size(400,400), 0, 0, cv::INTER_NEAREST);
		imshow("plot_cofusion - all", plot_cofusion);
		
		for(unsigned int i=0;i<vcap.size();i++){
			for(unsigned int r=0;r<frust_plots[i].size();r++){
				cm_plot_tdp(plot,vcm[i],frust_plots[i][r],Scalar(0,255,0));
			}
		}
		for(unsigned int i=0;i<vcap.size();i++){
			for(unsigned int r=0;r<mark_plots[i].size();r++){
				cm_plot_tdp(plot,vcm[i],mark_plots[i][r],Scalar(255,0,0));
			}
		}
		/*
		for(unsigned int i=0;i<vcap.size();i++){ //each camera
			for(unsigned int r=0;r<obj_plots[i].size();r++){ //each object
				for(unsigned int e=0;e<obj_plots[i][r].size();e++){ //every point in AABB
					cm_plot_tdp(plot,vcm[i],obj_plots[i][r][e],Scalar(255,0,0));
				}
			}
		}*/
		
		pl_draw_mark(plot);
		imshow("plot - all", plot);
		
		
        int key = waitKey(1);
        if(key>0){
			cout << key << endl;
		}
        if(key==27){
			
			for(unsigned int i=0;i<vcap.size();i++){
				
				char numstr[512]; // enough to hold all numbers up to 64-bits
				sprintf(numstr,"cap_image/cam%d_conf%d.png",vcid[i],vconf[i]);
				imwrite( numstr, vimg[i] );
				
				vcap[i].release();
			}
			
			return ;
		}
		
		ovimg=vimg;
		
    }
	 
	for(int i=0;i<vcap.size();i++){
		vcap[i].release();
	}
}

int main(int argc,char *argv[])
{
	
if(0){
	vector<ConvNet<FP>* > cnl;
	
	for(int q=1;q<=1;q++){
		char numstr[512]; // enough to hold all numbers up to 64-bits
		sprintf(numstr,"cifar100_v1/conv_no%d/cnet",q);
		string convpath(numstr);
		ConvNet<FP>* cnet=new ConvNet<FP>("");
		cnet->load(convpath);
		cnl.push_back(cnet);
	}
	
do_conv(cnl,do_rotate(do_pyramid(load_testimage())));
return 0;
}



	//load_testimage();
	
	
	//load_cifar100();
	for(int i=0;i<argc;i++){
		cout << argv[i] << endl;
	}
	

   cv::VideoCapture temp_camera;
   int maxTested = 10;
   for (int i = 0; i < maxTested; i++){
     cv::VideoCapture temp_camera(i);
     bool res = (temp_camera.isOpened());
     temp_camera.release();
     if (res)
     {
       cout << " Cam " << i << " is avaliable. " << endl;  
     }
   }
   
	
	vector<int> vcam;
	vcam.push_back(0);
	vcam.push_back(10);
	vcam.push_back(10);
	vcam.push_back(10);
	//vcam.push_back(10);
	
	
if(1){
	Mat asad;
	//selfcalib_dummy(asad);
	selfcalib(vcam);
	
	
	
	//calibrateCamera(1,10);
	//unDist(1,10);
	return 0;
}
	//for(;;);
	
	//Test Utils
	/*
	Utils<FP> ut;
	
	cout << "mrand" << ut.mrand() << endl;
	cout << "randf" << ut.randf(-1,1) << endl;
	cout << "randi" << ut.randi(-1,1) << endl;
	cout << "randn" << ut.randn(-1,1) << endl;
	cout << "guassRandom" << ut.gaussRandom() << endl;
	vector<FP> vf=ut.zeros(5);
	cout << vf.size() << endl;
	for(int i=0;i<vf.size();i++){
		cout << vf[i] << " ";
	}
	cout << endl;
	
	cout << "contains 0 : " << ut.arrContains(vf,0) << endl;
	cout << "contains 1 : " << ut.arrContains(vf,1) << endl;
	
	vector<FP> uvf=ut.arrUnique(vf);
	cout << uvf.size() << endl;
	for(int i=0;i<uvf.size();i++){
		cout << uvf[i] << " ";
	}
	cout << endl;
	
	
	for(int i=0;i<vf.size();i++){
		vf[i]=i;
		cout << vf[i] << " ";
	}
	cout << endl;
	
	map<string,FP> m=ut.maxmin(vf);
	cout << m["maxi"] << endl;
	cout << m["maxv"] << endl;
	cout << m["mini"] << endl;
	cout << m["minv"] << endl;
	cout << m["dv"] << endl;
	
	vector<int> vp=ut.randperm(10);
	for(int i=0;i<10;i++)
		cout << vp[i] << " ";
	cout << endl;
	*/
//ConvNet
	//Test Vol
	Vol<float>* v1=new Vol<float>(5,4,3);
	for(int i=0;i<5*4*3;i++){
		cout << v1->w[i] << " ";
	}
	cout << endl;
	
	delete v1;
	
	Vol<float>* v2=new Vol<float>(5,4,3,-1.3f);
	for(int i=0;i<5*4*3;i++){
		cout << v2->w[i] << " ";
	}
	cout << endl;
	
	delete v2;
	
	Mat image,gray_image;
    image = imread(getTrainingSetPath(0,0), CV_LOAD_IMAGE_COLOR);
    
    
    
    
    //Vol mat_to_img
    
    

		
    
    FP* pred=new FP[500];
	int i_pred=0;
	
	int* fcount=new int[100];
    
    int k=0;
    


    



    
string convpath="cifar100/conv_no1/cnet";
ConvNet<FP>* cnet=new ConvNet<FP>("");
/*
 * input[sx:32,sy:32,depth:3]>conv[sx:4,filters:40,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>conv[sx:4,filters:30,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>fc[num_neurons:60]>sigmoid[]>fc[num_neurons:10]>softmax[]

input[sx:32,sy:32,depth:3]>conv[sx:4,filters:30,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>conv[sx:4,filters:20,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>conv[sx:4,filters:20,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>fc[num_neurons:60]>sigmoid[]>fc[num_neurons:50]>sigmoid[]>fc[num_neurons:10]>softmax[]

\
input[sx:32,sy:32,depth:3]>conv[sx:4,filters:40,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>conv[sx:4,filters:50,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>conv[sx:4,filters:60,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>fc[num_neurons:10]>softmax[]\

*/

Utils<FP> ut;
FP rp=FP(0);
int saw=0;

	vector<Mat> vm = load_cifar100(); //load_cifar100();//
	vector<int> vl = load_cifar100_label(); //load_cifar100_label();//
cout << "Loading" << endl;
//cnet->save(convpath);
cnet->load(convpath);
cout << "Fin Loading" << endl;

cnet->learning_rate=FP(0.04);
cnet->batch_size=6;
cnet->method = "sgd";
//sgd/adagrad/adadelta/windowgrad/netsterov
cnet->momentum=FP(0.5);

int ntest=100;
int nclass=20;
int ntrain=vm.size();

vector<string> lbs;
	//'airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
 lbs.push_back("airplane");
 lbs.push_back("car");
 lbs.push_back("bird");
 lbs.push_back("cat");
 lbs.push_back("deer");
 lbs.push_back("dog");
 lbs.push_back("frog");
 lbs.push_back("horse");
 lbs.push_back("ship");
 lbs.push_back("truck");
cnet->par_init(10);
cnet->par_sync();

    while(true)
    {
		
		{
			cout << "init" << endl;
			 //7 cores
			vector<int> ks(100);
			
			int slc=10;
			#pragma omp parallel for
			for(int q=0;q<7;q++){
				for(int w=0;w<slc;w++){
					ks[q] = (int)(  ut.mrand() * (vm.size())  );
					if(ut.mrand()<0.5){
						flip(vm[ks[q]],vm[ks[q]],1);
					}
					Vol<FP>* vx3 = Vol<FP>::mat_to_vol(vm[ks[q]]);
					
					cnet->par_forward(q,vx3);
					FP tpb = cnet->par_getProb(q);
					
					if( vl[ks[q]] == cnet->par_getPrediction(q) && tpb > 0.5 ){
					}
					else{
						cnet->par_train(q,vx3,vl[ks[q]]);
					}
					
					
					delete vx3;
				}
			}
			#pragma omp barrier
			cout << " Start Merge" << endl;
			cnet->par_mergegrad();
			cout << " End Merge" << endl;
			
			cout << " End Dtor" << endl;
			//force train
			cnet->updatew(7*slc);
			cnet->par_sync();
		}
		saw++;
	{
	
	//#pragma omp parallel for
	delete fcount;
	fcount=new int[200];
	ntest=7*20;
	#pragma omp parallel for
			for(int q=0;q<7;q++){
				for(int w=0;w<20;w++){
					int kk=  (int)(  ut.mrand() * (vm.size())  );
					//kk+=(vm.size()-1000);
					Vol<FP>* v4 = Vol<FP>::mat_to_vol(vm[kk]);
					cnet->par_forward(q,v4);
					FP result = ( vl[kk] == cnet->par_getPrediction(q) && cnet->par_getProb(q) > 0.5 )?FP(1.0):FP(0.0);
					
					if( vl[kk] == cnet->par_getPrediction(q) ){
							result = FP(1.0);
						}
						else{
							result = FP(0.0);
							#pragma omp atomic
							fcount[vl[kk]]++;
						}
					#pragma omp critical
					{
					pred[i_pred]= result;
					i_pred++;
					}
					i_pred = (i_pred >= ntest)?0:i_pred;
					delete v4;
				}
			}
	#pragma omp barrier
	//#pragma omp barrier
	
		for(int i=0;i<ntest;i++){
			rp+=pred[i];
		}
		rp/=ntest;
		
		
	
		
	}
	cout << " saw : " << saw << "  Correct Percent : " << rp << endl;
	if(saw%1==0){
		std::ofstream outfile;
		outfile.open(convpath, std::ios_base::app);
		outfile <<  rp << endl; 

		cout << convpath << endl;
		cnet->save(convpath);
		cnet->load(convpath);
	}
		//cvtColor(image, gray_image, CV_BGR2GRAY);
		
		//>pool[sx:5,sy:5]>fc[num_classes:10]
		
//>softmax[num_classes:10]
		
		
		continue;
		cnet->par_destruct();
		
		
		Vol<FP>* v3 = Vol<FP>::mat_to_vol(vm[k]);
		//static int pp=0;
		//if(pp++==0)
		cnet->forward(v3);
		FP result = ( vl[k] == cnet->getPrediction() )?FP(1.0):FP(0.0);
		//if(result < 0.5)
		
		saw++;
		
		//cout << "Result " << vl[k] << cnet->getPrediction() << endl;
	if(saw%100==0){
	
	//#pragma omp parallel for
	delete fcount;
	fcount=new int[100];
	
	for(int q=0;q<ntest;q++){
		int kk=  (int)(  ut.mrand() * (vm.size())  );
		//kk+=(vm.size()-1000);
		Vol<FP>* v4 = Vol<FP>::mat_to_vol(vm[kk]);
		cnet->forward(v4);
		FP result = ( vl[kk] == cnet->getPrediction() )?FP(1.0):FP(0.0);
		
		if( vl[kk] == cnet->getPrediction() ){
				result = FP(1.0);
			}
			else{
				result = FP(0.0);
				fcount[vl[kk]]++;
			}
		
		pred[i_pred++]= result;
		i_pred = (i_pred >= ntest)?0:i_pred;
		delete v4;
	}
	//#pragma omp barrier
	
		for(int i=0;i<ntest;i++){
			rp+=pred[i];
		}
		rp/=ntest;
		
		
	
		
	}
	
	if(saw%500==0){
		cout << convpath << endl;
		cnet->save(convpath);
		cnet->load(convpath);
	}
	
		cout << k << "/" << vm.size() << " saw : " << saw << "  Correct Percent : " << rp << endl;
		//Vol<FP>* v4 = cnet->forward(v3);
		//Mat convnet = v4->npho_to_mat();
		//cout << "!!!" << endl;
		for(int i=0;i<cnet->net.size();i++){
			
			//cout << cnet->net[i]->get_layer_type() << " " << cnet->net[i]->get_out_act()->sx  << " " << cnet->net[i]->get_out_act()->sy << " "  << cnet->net[i]->get_out_act()->depth << endl;
			Mat inp = cnet->net[i]->get_out_act()->npho_to_mat();
			if(i==cnet->net.size()-1)
			inp = cnet->net[i]->get_out_act()->po_to_mat();
			//cout << " == " << inp.cols << " " << inp.rows << endl;
			resize(inp,inp,Size(512,512), 0, 0, INTER_AREA);
			char numstr[512]; // enough to hold all numbers up to 64-bits
			sprintf(numstr,"%d %s",i,cnet->net[i]->get_layer_type().c_str());
			imshow(numstr , inp );
		}
		//cout << "@@@" << endl;
		//if(result > 0.5)
		k=  (int)(  ut.mrand() * (vm.size()-1000)  );
		
		int imin=9;
		int vmin=0;
		for(int i=0;i<nclass;i++){
			if(vmin<fcount[i]){
				vmin=fcount[i];
				imin=i;
			}
		}
		
		//cout << "###" << endl;
		
		if(0&&ut.mrand()<0.1){
			cout << "Fault " << imin << endl;//" " << lbs[imin] << endl;
			while(vl[k]!=imin){
				k=  (int)(  ut.mrand() * (vm.size()-1000)  );
				//k++;
			}
		}
		
		if(k>vm.size()-1)
			k=0;
			
		delete v3;
		//delete v4;

        int key = waitKey(1);
        if(key==27)
            return 0;
    }
    
	delete cnet;
	
	
	
	/*
	int numBoards = 40;
    int numCornersHor = 4;
    int numCornersVer = 3;
    
    picojson::value v;
    v.value();
    
    printf("Enter number of corners along width: ");
    scanf("%d", &numCornersHor);

    printf("Enter number of corners along height: ");
    scanf("%d", &numCornersVer);

    printf("Enter number of boards: ");
    scanf("%d", &numBoards);
    
    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor, numCornersVer);
    VideoCapture capture = VideoCapture(0);
	vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > image_points;
    vector<Point2f> corners;
    int successes=0;
    
    Mat image;
    Mat gray_image;
    capture >> image;
    
    vector<Point3f> obj;
    for(int j=0;j<numSquares;j++)
        obj.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f));
        
    while(successes<numBoards)
    {
		cvtColor(image, gray_image, CV_BGR2GRAY);
		bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if(found)
        {
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray_image, board_sz, corners, found);
        }
        imshow("win1", image);
        
        char winstr[21]="win2"; // enough to hold all numbers up to 64-bits
        char numstr[21]; // enough to hold all numbers up to 64-bits
        sprintf(numstr,"%d",successes);
		strcat( winstr ,numstr);

        imshow( winstr , gray_image);

        capture >> image;
        int key = waitKey(1);
        
        if(key==27)

            return 0;

        if(key==' ' && found!=0)
        {
            image_points.push_back(corners);
            object_points.push_back(obj);

            cout << "Snap stored!" << successes;

            successes++;

            if(successes>=numBoards)
                break;
        }
	}
	
	Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
    
    
    Mat imageUndistorted;
    while(1)
    {
        capture >> image;
        undistort(image, imageUndistorted, intrinsic, distCoeffs);

        imshow("win1", image);
        imshow("win2", imageUndistorted);
        waitKey(1);
    }
	capture.release();

    return 0;
	*/
	
	/*
    help();

    string fileName = "cube4.avi";
    VideoCapture video(fileName);
    if (!video.isOpened())
    {
        cerr << "Video file " << fileName << " could not be opened" << endl;
        return EXIT_FAILURE;
    }

    Mat source, grayImage;
    video >> source;

    namedWindow("Original", WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
    namedWindow("POSIT", WINDOW_OPENGL | CV_WINDOW_FREERATIO);
    resizeWindow("POSIT", source.cols, source.rows);

    displayOverlay("POSIT", "We lost the 4 corners' detection quite often (the red circles disappear).\n"
                   "This demo is only to illustrate how to use OpenGL callback.\n"
                   " -- Press ESC to exit.", 10000);

    float OpenGLMatrix[] = { 1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1 };
    setOpenGlContext("POSIT");
    setOpenGlDrawCallback("POSIT", on_opengl, OpenGLMatrix);

    vector<CvPoint3D32f> modelPoints;
    initPOSIT(&modelPoints);

    // Create the POSIT object with the model points
    CvPOSITObject* positObject = cvCreatePOSITObject( &modelPoints[0], (int)modelPoints.size());

    CvMatr32f rotation_matrix = new float[9];
    CvVect32f translation_vector = new float[3];
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1e-4f);
    vector<CvPoint2D32f> srcImagePoints(4, cvPoint2D32f(0, 0));

    while (true)
    {
        video >> source;
        if (source.empty())
            break;

        imshow("Original", source);

        //foundCorners(&srcImagePoints, source, grayImage);
        //cvPOSIT(positObject, &srcImagePoints[0], FOCAL_LENGTH, criteria, rotation_matrix, translation_vector);
        //createOpenGLMatrixFrom(OpenGLMatrix, rotation_matrix, translation_vector);

        updateWindow("POSIT");
        int keycode=waitKey(33);
		if(keycode>-1){
			cout << keycode << endl;
		}
        if (video.get(CV_CAP_PROP_POS_AVI_RATIO) > 0.99)
            video.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
    }

    setOpenGlDrawCallback("POSIT", NULL, NULL);
    destroyAllWindows();
    cvReleasePOSITObject(&positObject);

    delete[]rotation_matrix;
    delete[]translation_vector;

    return EXIT_SUCCESS;*/
}
