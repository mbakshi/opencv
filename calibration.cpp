#include <stdio.h>
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <sstream>
#include <time.h>

#define DEPTH_SENSOR		2
#define RGB_SENSOR			0
#define EXT_SENSOR			1
#define NUM_COLOR_SENSORS	2
#define NUM_SENSORS			3

using namespace cv;
using namespace std;

// Interpolate color of a point with non-integer coordinates in an image
// You can use this function to get smoother outputs, but it is optional
Vec3b avSubPixelValue8U3( const Point2f pt, const Mat img )
{
	int floorx = (int)floor( pt.x );
	int floory = (int)floor( pt.y );

	if( floorx < 0 || floorx >= img.cols-1 || 
		floory < 0 || floory >= img.rows-1 )
		return 0;

	float px = pt.x - floorx;
	float py = pt.y - floory;

	Vec3b tl = img.at<Vec3b>(floory,floorx);
	Vec3b tr = img.at<Vec3b>(floory,floorx+1);
	Vec3b bl = img.at<Vec3b>(floory+1,floorx);
	Vec3b br = img.at<Vec3b>(floory+1,floorx+1);
	Vec3b result;
	for (int i=0;i<3;i++)
		result[i] = (unsigned char)floor(tl[i] * (1-px)*(1-py) + tr[i]*px*(1-py) + bl[i]*(1-px)*py + br[i]*px*py + 0.5 );

	return result;
}

// Load the depth image, the RGB & the external image for a specific frame
// If all images can be loaded, return true
// Otherwise, return false
bool loadImages(char* src, int frame, Mat& depthMat, Mat& rgbMat, Mat& extMat){
	char fname[10];
	char fpath[100];

	// load depth image
	printf("Frame %04d\n",frame);
	sprintf(fname,"%04d.png",frame);
	sprintf(fpath,"%s/Depth/%s",src,fname);
	depthMat = imread(fpath, CV_LOAD_IMAGE_ANYDEPTH);
	if (!depthMat.data ) {
		return false;
	}
	
	// load RGB image
	sprintf(fpath,"%s/RGB/%s",src,fname);
	rgbMat = imread(fpath);
	if (!rgbMat.data) {
		return false;
	}
	
	// load external image
	sprintf(fpath,"%s/Ext/%s",src,fname);
	extMat = imread(fpath);
	if (!extMat.data) {
		return false;
	}
	// TODO 0: In the sample dataset, the external images need to be filipped vertically to have the same direction with the others
	// In your own dataset, however, you may not need to do it. Check your images and comment out the statement below if not needed.
	flip(extMat,extMat,1);
	return true;
}

// Extract board information from the input file
bool getboardInfor(char* boardInforPath, Size &boardSize, float &squareSize, Rect_<float> &boardRegion){
	FILE* file = fopen(boardInforPath, "r");
	if (!file) {
		fprintf(stderr,"Error! Can't find the board information file!\n");
		return  false;
	}

	float right, bottom;
	fscanf(file, "%d %d %f %f %f %f %f",&boardSize.width,&boardSize.height,&squareSize, &boardRegion.x, &boardRegion.y, &right, &bottom);
	boardRegion.width = right - boardRegion.x;
	boardRegion.height = bottom - boardRegion.y;
	fclose(file);
	return true;
}

// Compute 3-D coordinates of the checker inner corners in {B}
// Input:
//		boardSize : checker size
//		squareSize: side of each cell in the checker pattern
// Output:
//		inCorners3D: list of computed 3-D coordinates
//		matInCorners3D: matrices 4xN of computed 3-D homogeneous coordinates
//
void calcInnerCorner(Size boardSize, float squareSize, vector<Point3f>& inCorners3D, Mat &matInCorners3D)
{
	// TODO I.1.a: Compute 3-D coordinates of the checker inner corners in {B}
	//             Remember to list them in order from top-left corner to the right-bottom one
	//
	// fill your code here
	//
	matInCorners3D = Mat::zeros(3, 20, CV_32F);
	int matitr=0;
	for( int j = 0; j < boardSize.height; j++ )
		for( int i = 0; i < boardSize.width; i++ )  
			{
                inCorners3D.push_back(Point3f(float((i-2)*squareSize),
                                          float((j*squareSize)-52.5), 0));
				float FourDpoints[3]={float((i-2)*squareSize),
					float((j*squareSize)-52.5), 0};
				Mat(3, 1, CV_32F, &FourDpoints).copyTo(matInCorners3D.col(matitr++));
				
			}
			//convertPointsToHomogeneous(inCorners3D,hom);
			//matInCorners3D=Mat(hom);
	

				//	Mat.row(j).col(i)=inCorners3D
	cout<<1<<endl;
       
}

// Compute 3-D coordinates of the board outer corners in {B}
// Input:
//		boardRegion
// Output:
//		boardCorners3D
//		matBoardCorners3D
//
void calcOuterCorner(Rect_<float> boardRegion, vector<Point3f>& boardCorners3D, Mat &matBoardCorners3D)
{
	// TODO I.1.b: Compute 3-D coordinates of the board 4 outer corners in {B}
	//
	// fill your code here
	//(-150, -103.5, 0), (138, -103.5, 0), (138, 110.5, 0) and (-150, 110.5, 0).

	boardCorners3D.push_back(Point3f(float(-150),float(-103.5), 0));
	boardCorners3D.push_back(Point3f(float(138),float(-103.5), 0));
	boardCorners3D.push_back(Point3f(float(-150),float(110.5), 0));
	boardCorners3D.push_back(Point3f(float(138),float(110.5), 0));
	matBoardCorners3D = Mat::zeros(3, 4, CV_32F);

	matBoardCorners3D.at<float>(0, 3) = -150;
	matBoardCorners3D.at<float>(1, 3) = -103.5;
	matBoardCorners3D.at<float>(0, 0) = 138;
	matBoardCorners3D.at<float>(1, 0) = -103.5;
	matBoardCorners3D.at<float>(0, 1) = -150;
	matBoardCorners3D.at<float>(1, 1) = 110.5;
	matBoardCorners3D.at<float>(0, 2) = 138;
	matBoardCorners3D.at<float>(1, 2) = 110.5;
}

// calibrate a camera using Zhang's method
bool runCalibration( vector<Point3f> inCorners3D, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs, 
	vector<vector<Point2f> > incorner2DPoints, vector<Mat>& rvecs, vector<Mat>& tvecs)
{
	// TODO II.1.2: Calibrate the camera using function calibrateCamera of OpenCV
	//              Choose suitable flags 
	//
	// fill your code here
	vector<vector<Point3f>> inCorner3DVec;
	for (int itr = 0; itr < incorner2DPoints.size(); ++itr)
		inCorner3DVec.push_back(inCorners3D);
	//points should be 3D because they are in a pattern corrdinate system
	//cout<<distCoeffs<<endl;
	//cout<<cameraMatrix<<endl;
	//getchar();
	calibrateCamera(inCorner3DVec, incorner2DPoints,imageSize, cameraMatrix, distCoeffs, rvecs,  tvecs);
	//cout<<distCoeffs<<endl;
	//cout<<cameraMatrix<<endl;
	//getchar();

	//
	return true;
}

// Find a non-zero point from a starting point in a thresholded image
// Input:
//		img    : the thresholded image
//		start  : starting point
//		maxDist: maximum distance to search
// Output:
//		dst    : a non-zero point
// If no point is found, return false
bool findClosestPoint(Mat img, Point2f start, Point2f &dst, int maxDist = 10){
	int x = floor(start.x + 0.5);
	int y = floor(start.y + 0.5);
	if (img.at<unsigned char>(y,x) > 0){
		dst = start;
		return true;
	}
	for (int i=1;i<=maxDist;i++){
		if (x - i >= 0){				// Search on the left
			for (int j=-i;j<=i;j++){
				if (y + j < 0 || y + j > img.rows || img.at<unsigned char>(y + j,x-i) == 0) continue;
				dst.x = x -i;
				dst.y = y + j;
				return true;
			}
		}
		
		if (x + i < img.cols){			// Search on the right
			for (int j=-i;j<=i;j++){
				if (y + j < 0 || y + j > img.rows || img.at<unsigned char>(y + j,x+i) == 0) continue;
				dst.x = x + i;
				dst.y = y + j;
				return true;
			}
		}
		
		if (y - i >= 0){				// Search upward
			for (int j=-i;j<=i;j++){
				if (x + j < 0 || x + j > img.cols || img.at<unsigned char>(y - i,x+j) == 0) continue;
				dst.x = x + j;
				dst.y = y - i;
				return true;
			}
		}
		
		if (y + i < img.rows){			// Search downward
			for (int j=-i;j<=i;j++){
				if (x + j < 0 || x + j > img.cols || img.at<unsigned char>(y + i,x+j) == 0) continue;
				dst.x = x + j;
				dst.y = y + i;
				return true;
			}
		}
	}
	dst = start;
	return false;
}

// Write the pointcloud extracted from a depth image & its colored image to PLY file
// Input:
//		count  : the number of points in the cloud
//		f      : focal length of the depth sensor
//		color  : the colored depth image
//		fname  : file name
void writePLY(int count, float f, Mat color, Mat depth, char* fname){
	FILE* file = fopen( fname, "w");

	if ( !file )
    {
		std::cerr << "Creation Error\n";
        return;
    }

	fprintf( file, "ply\n");
	fprintf( file, "format ascii 1.0\n" );
	fprintf( file, "element vertex %d\n", count );
	fprintf( file, "property float x\nproperty float y\nproperty float z\n" );
	fprintf( file, "property uchar blue\nproperty uchar green\nproperty uchar red\n");
	fprintf( file, "end_header\n");

	for (int i=0;i<depth.cols;i++){
		for (int j=0;j<depth.rows;j++){
			if (depth.at<short>(j,i) > 0){
				float Z = depth.at<short>(j,i);
				float X = (i-depth.cols/2) * Z/f;
				float Y = (j-depth.rows/2) * Z/f;
				Vec3b colors = color.at<Vec3b>(j,i);
				fprintf( file, "%f %f %f %d %d %d\n",X, Y, Z, colors[0], colors[1], colors[2]);
			}
		}
	}
	fclose(file);
}

int main(int argc, char** argv){
	char src[80] = "../Data";					// path to the dataset
	char dst[80] = "../Output";				// path to the output directory
	char boardInfor[80] = "../Data/board.txt";  // the board information file

	int start = 40;								// start frame
	int end = 210;								// the last frame can be accessed in the dataset
	int step = 1;								// step in frame numbers to process
	int nFrame2Use = 10;						// number of frames can be used to calibrate
	Rect_<float> boardRegion;		// boardRegion

	vector<int> goodFrames;						// list of good frames to use for calibration
	Size boardSize;
	boardSize.width = 5;
	boardSize.height = 4;
	float squareSize = 35;
	float fc = 285.171;
	float fd = 285.171;

	int arg = 0;								// Process input parameters
	while( ++arg < argc) 
	{ 
		// Input directory
		if( !strcmp(argv[arg], "-i") )
			strcpy(src, argv[++arg] );
		
		// Output directory
		if( !strcmp(argv[arg], "-o") )
			strcpy(dst, argv[++arg] );

		// First frame
		if( !strcmp(argv[arg], "-b") )
			start = atoi( argv[++arg] );
		
		// Step
		if( !strcmp(argv[arg], "-s") )
			step = atoi( argv[++arg] );

		// Last frame
		if( !strcmp(argv[arg], "-e") )
			end = atoi( argv[++arg] );

		// Number of frames to calibrate
		if( !strcmp(argv[arg], "-n") )
			nFrame2Use = atoi( argv[++arg] );

		// Focal length
		if( !strcmp(argv[arg], "-fc") )
			fc = atof( argv[++arg] );
		
		// Focal length
		if( !strcmp(argv[arg], "-fd") )
			fd = atof( argv[++arg] );

		// Board information file
		if( !strcmp(argv[arg], "-v") )
			strcpy(boardInfor, argv[++arg] );
	}

	// get board information from file
	getboardInfor(boardInfor, boardSize, squareSize, boardRegion);
	
	bool gotSize = false;
	bool extCalibrated = false;
	Size imageSize[NUM_SENSORS];
	Mat cameraMatrix[NUM_SENSORS];										// cameras' intrinsic matrices
	vector<Mat> rvecs[NUM_COLOR_SENSORS], tvecs[NUM_COLOR_SENSORS];		// board pose regarding 2 color cameras
	vector<vector<Point2f> > incorner2DPoints[NUM_COLOR_SENSORS];		// extracted inner corners
	vector<vector<Point2f> > boardCorner2DPoints;						// extracted board outer corners from depth images

	vector<float> reprojErrs[NUM_SENSORS];
	vector<Point3f> inCorners3D;											// 3D coordinates of inner corners in {B}
	Mat matInCorners3D;
	vector<Point3f> boardCorners3D;										// 3D coordinates of the board outer corners in {B}
	Mat matBoardCorners3D;
	vector<vector<short>> allBoardCornerDepth;

	// prepared reference 3-D coordinates
	calcInnerCorner(boardSize, squareSize, inCorners3D, matInCorners3D);
	calcOuterCorner(boardRegion, boardCorners3D,matBoardCorners3D);
	
	
	// Initiate camera matrices
	float _cam[9] = {fc, 0 , 0, 0, fc, 0, 0 , 0, 1};
	cameraMatrix[RGB_SENSOR] = Mat(3,3,CV_32F,_cam);
	float _extCam[9] = {0, 0 , 0, 0, 0, 0, 0 , 0, 1};
	cameraMatrix[EXT_SENSOR] = Mat(3,3,CV_32F,_extCam);	
	float _depthcam[9] = {fd, 0 , 0, 0, fd, 0, 0 , 0, 1};
	cameraMatrix[DEPTH_SENSOR] = Mat(3,3,CV_32F,_depthcam);
	// null distortation parameters
	Mat distCoeffs(1,4,CV_32F);
	distCoeffs = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////   STEP 1: Scan over images & collect useful information    ////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////
	char fpath[100];
	for (int frame=start;frame<=end;frame+=step){
		Mat depthMat, rgbMat, extMat;

		vector<Point2f> rgbPointBuf;			// temporary extracted 2-D inner corners
		vector<Point2f> extPointBuf;

		/// Load images & validate
		if (!loadImages(src, frame, depthMat, rgbMat, extMat)){
			continue;
		}

		if (!gotSize) {
			// get image sizes
			imageSize[RGB_SENSOR].height = rgbMat.rows;
			imageSize[RGB_SENSOR].width  = rgbMat.cols;
			imageSize[DEPTH_SENSOR].height = depthMat.rows;
			imageSize[DEPTH_SENSOR].width  = depthMat.cols;
			imageSize[EXT_SENSOR].height = extMat.rows;
			imageSize[EXT_SENSOR].width  = extMat.cols;

			// set cameras' principal points
			_cam[2] = imageSize[RGB_SENSOR].width/2;
			_cam[5] = imageSize[RGB_SENSOR].height/2;
			_extCam[2] = imageSize[EXT_SENSOR].width/2;
			_extCam[5] = imageSize[EXT_SENSOR].height/2;
			_depthcam[2] = imageSize[DEPTH_SENSOR].width/2;
			_depthcam[5] = imageSize[DEPTH_SENSOR].height/2;
			gotSize = true;
		}

		//////  Process the color images
		// TODO II.1.1: Extract inner corners of the checkerboard (extPointBuf) from the external image (extMat)
		//              If no corner is detected, skip this frame
		//			    If the detected corners are in a wrong order, revert it
		//
		// fill your code here
		//
		
		bool find=findChessboardCorners( extMat, boardSize, extPointBuf,
                    CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		if(extPointBuf[0].x > extPointBuf[19].x)
			reverse(extPointBuf.begin(),extPointBuf.end());
		if(!find) continue;


		if (extPointBuf.size() > 0)
			drawChessboardCorners( extMat, boardSize, Mat(extPointBuf), true );

		//resize extMat
		Mat extResized;
		resize(extMat,extResized,rgbMat.size());
		cv::namedWindow( "External image", 10 );
		cv::imshow( "External image", extResized );
		cv::waitKey( 20 );
		// TODO I.2.1: Extract inner corners of the checkerboard (extPointBuf) from the external image (extMat)
		//             If no corner is detected, skip this frame
		//			   If the detected corners are in a wrong order, revert it
		//
		// fill your code here
		//
		 find=findChessboardCorners( rgbMat, boardSize, rgbPointBuf,
                    CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		// cout<<rgbPointBuf<<endl;
		// getchar();
		 if(rgbPointBuf[0].x > rgbPointBuf[19].x)
			reverse(rgbPointBuf.begin(),rgbPointBuf.end());
		if(!find) continue;


		if (rgbPointBuf.size() > 0)
			drawChessboardCorners( rgbMat, boardSize, Mat(rgbPointBuf), true );
		cv::namedWindow( "RGB image", 11 );
		cv::imshow( "RGB image", rgbMat );
		cv::waitKey( 20 );

		Mat rVec, tVec;
		// TODO I.2.2: From the detected 2D inner corners (rgbPointBuf) & their 3D coordinates in {B} (inCorners3D), estimate the board pose 
		//             (regarding the color sensor coordinate system {C}) using function solvePnPRansac() of OpenCV. 
		//			   Save your result into rVec and tVec for later use
		//
		// fill your code here
		solvePnPRansac(inCorners3D, rgbPointBuf, cameraMatrix[RGB_SENSOR], distCoeffs,  rVec,  tVec);
	//	cout<<inCorners3D<<"\n\n\n";
	//	cout<<rgbPointBuf<<"\n\n\n";
	//	cout<<rVec<<endl<<"\n\n\n";
	//	cout<<tVec<<endl<<"\n\n\n";
	//	getchar();
		//

		vector<Point2f> rgbPoints2;
		projectPoints(inCorners3D,rVec,tVec,cameraMatrix[RGB_SENSOR],distCoeffs,rgbPoints2);	// back-project inner corners into the RGB image
		for (int i=0;i<rgbPoints2.size();i++)
			circle(rgbMat,Point(floor(rgbPoints2[i].x + 0.5),floor(rgbPoints2[i].y + 0.5)),2,Scalar(128,128,0),2);
	    cv::imshow( "RGB image", rgbMat );
		cv::waitKey( 20 );

		rVec.convertTo(rVec, CV_32F);
		tVec.convertTo(tVec, CV_32F);

		//////  Process the depth image
		Mat depthMat0 = depthMat / 4;					// scale depth map
		depthMat0.convertTo(depthMat0, CV_8U);			// gray-scale depth image

		// duplicate channels
		Mat depthMat1(depthMat.rows,depthMat.cols,CV_8UC3);	// true-color depth image
		insertChannel(depthMat0,depthMat1,0);
		insertChannel(depthMat0,depthMat1,1);
		insertChannel(depthMat0,depthMat1,2);

		cv::namedWindow( "Depth image", 12 );		
		cv::imshow( "Depth image", depthMat1 );
		cv::waitKey( 20 );
		
		// plot locations of the inner corners (from the RGB image) onto the depth image
		for (int i=0;i<boardSize.height;i++){
			for (int j=0;j<boardSize.width;j++){
				circle(depthMat1,Point(floor(rgbPointBuf[i*boardSize.width +j].x),floor(rgbPointBuf[i*boardSize.width +j].y)),2,Scalar(0,0,255));
			}
		}
		cv::imshow( "Depth image", depthMat1 );
		cv::waitKey( 20 );
		
		Mat segmentedDepthMat;
		// TODO I.2.3.a: Detect the board region using a segmentation technique (thresholding, watershed, mean-shift ...). 
		//					Input: the gray-scale depth map (depthMat0)
		//					Output: a depth map with only the extracted board region (segmentedDepthMat)
		//               Note that you can get hints about the board position from the inner corners location in the color image.
		//
		// fill your code here

		uchar min=depthMat0.at<uchar>(rgbPointBuf[0].y,rgbPointBuf[0].x);
		uchar max=depthMat0.at<uchar>(rgbPointBuf[19].y,rgbPointBuf[19].x);
		for(size_t i=0; i < rgbPointBuf.size(); i++)
		{
			if(depthMat0.at<uchar>(rgbPointBuf[i].y,rgbPointBuf[i].x)<min)
				min=depthMat0.at<uchar>(rgbPointBuf[i].y,rgbPointBuf[i].x); //find the min depth
			if(depthMat0.at<uchar>(rgbPointBuf[i].y,rgbPointBuf[i].x)>max)
				max=depthMat0.at<uchar>(rgbPointBuf[i].y,rgbPointBuf[i].x);  //find the max depth

		}
		min=(min-15)>0?min-15:0;
		max=(max+15)>0?max+15:0; //keeping a constant margin
	//	cout<<min<<"\t"<<max;
		
		threshold( depthMat0, segmentedDepthMat, min , 1, 3 );
		threshold( segmentedDepthMat, segmentedDepthMat, max , 1, 4 );
	//	cv::imshow( "Segmented image", segmentedDepthMat);
	//	getchar();
		
		//
		
		vector<Vec4i> lines;
		// TODO I.2.3.b: From segmentedDepthMat, extract edges by Canny (or other) detector
		//				 Detect lines using HoughLinesP	
		//
		// fill your code here
		Canny( segmentedDepthMat, depthMat0, min/2, max + 10);
		//preserve segmentedDepthMap
		double h=abs(rgbPointBuf[19].y-rgbPointBuf[0].y);
		vector<Vec4i> tempEdges;
		HoughLinesP(depthMat0, tempEdges, 1, CV_PI/180, 0.81*h, 0.4*h, ((rgbPointBuf[0].y-rgbPointBuf[19].y) * 0.3>0?(rgbPointBuf[0].y-rgbPointBuf[19].y) * 0.3: 7.0));
		if(tempEdges.size()==0) continue;
		//cv::imshow( "depth", depthMat0);
		//cout<<tempEdges[0]<<endl;
	//	getchar();
		//0.75 * h --> threshhold
		//0.375 * h --> minimum acceptable length of a line
		
		bool boardEdgeFound[4] = {0};
		Vec4i boardEdges[4];
		// TODO I.2.3.c: Save detect lines into vector boardEdges.
		//               You can use the known inner corners to filter the detected lines to make sure that exactly 1 line is picked for each board edge 
		//
		// fill your code here
		//vector<Vec4i> tempEdges;
		//HoughLinesP(segmentedDepthMat, tempEdges, 1, CV_PI/180, 50 );
		//cout<<rgbPointBuf<<endl<<endl<<endl;
	//	for(size_t itr2=0; itr2<tempEdges.size();itr2++)
	//		cout<<tempEdges[itr2][0]<<", "<<tempEdges[itr2][1]<<", "<<tempEdges[itr2][2]<<", "<<tempEdges[itr2][3]<<endl;
	//	getchar();
		double max1=0, max2=0, max3=0, max4=0;
		for(size_t i=0;i<tempEdges.size();i++)
		{
			double tx1=tempEdges[i][0];
			double ty1=tempEdges[i][1];
			double tx2=tempEdges[i][2];
			double ty2=tempEdges[i][3];

			if(tx1<rgbPointBuf[0].x)
			{
				if(ty1<rgbPointBuf[0].y && ty2<rgbPointBuf[0].y)
				{
					if(max1==0)
					{
						max1=abs(tx1-tx2);
						boardEdges[0]=tempEdges[i];
						boardEdgeFound[0]=1;
					}
					else if(abs(tx1-tx2)>max1)
					{
						boardEdges[0]=tempEdges[i];
						max1=abs(tx1-tx2);
					}
				}
				else if(ty1>rgbPointBuf[16].y && ty2>rgbPointBuf[16].y)
				{
					if(max3==0)
					{
						max3=abs(tx1-tx2);
						boardEdges[2]=tempEdges[i];
						boardEdgeFound[2]=1;
					}
					else if(abs(tx1-tx2)>max3)
					{
						boardEdges[2]=tempEdges[i];
						max3=abs(tx1-tx2);
					}
				}
			}
			if(ty1<rgbPointBuf[0].x)
			{
				if(tx1<rgbPointBuf[0].x && tx2<rgbPointBuf[0].x)
				{
					if(max2==0)
					{
						max2=abs(ty1-ty2);
						boardEdges[1]=tempEdges[i];
						boardEdgeFound[1]=1;
						
					}
					else if(abs(ty1-ty2)>max2)
					{
						boardEdges[1]=tempEdges[i];
						max2=abs(ty1-ty2);
					}
				}
				else if(tx1>rgbPointBuf[4].x && tx2>rgbPointBuf[4].x)
				{
					if(max4==0)
					{
						max4=abs(ty1-ty2);
						boardEdges[3]=tempEdges[i];
						boardEdgeFound[3]=1;
					}
					else if(abs(ty1-ty2)>max4)
					{
						boardEdges[3]=tempEdges[i];
						max4=abs(ty1-ty2);
					}
				}
			}
			
			/*
			if(ty1 < rgbPointBuf[0].y && ty2 < rgbPointBuf[0].y )
			{
				boardEdges[0]=tempEdges[i];
				boardEdgeFound[0]=TRUE;
			}
			if(tx1 < extPointBuf[0].x && tx2 < extPointBuf[0].x )
			{
				boardEdges[1]=tempEdges[i];
				boardEdgeFound[1]=1;
			}
			if(ty1 > extPointBuf[extPointBuf.size()-1].y && ty2 > extPointBuf[extPointBuf.size()-1].y )
			{
				boardEdges[2]=tempEdges[i];
				boardEdgeFound[2]=1;
			}
			if(tx1 > extPointBuf[extPointBuf.size()-1].x && tx2 > extPointBuf[extPointBuf.size()-1].x )
			{
				boardEdges[3]=tempEdges[3];
				boardEdgeFound[3]=1;
			}
			*/
		}
				
		//

		bool allBorderFound = true;
		for (int i=0;i<4;i++){
			if (!boardEdgeFound[i]) {
				allBorderFound = false;
				break;
			}
			else
			{
				line(depthMat1,Point(boardEdges[i][0],boardEdges[i][1]),Point(boardEdges[i][2],boardEdges[i][3]),Scalar(0,0,255),2);
			}
		}
		
		if (!allBorderFound) continue;
		cv::imshow( "Depth image", depthMat1 );
		cv::waitKey( 20 );

		vector<Point2f> board2DCorners;
		// TODO I.2.3.d: Compute the board outer corners as the intersections of the board edges (in boardEdges)
		//				 Save them into board2DCorners
		//				 Make sure that your corners are sorted in the correct order regarding the computed 3-D coordinates in part I.1.b
		//
		// fill your code here
		// 1st point
		Point2f o1=Point2f(float(boardEdges[0][0]), float(boardEdges[0][1]));
		Point2f p1=Point2f(float(boardEdges[0][2]), float(boardEdges[0][3]));
		Point2f o2=Point2f(float(boardEdges[1][0]), float(boardEdges[1][1]));
		Point2f p2=Point2f(float(boardEdges[1][2]), float(boardEdges[1][3]));
		Point2f x = o2 - o1;
	    Point2f d1 = p1 - o1;
	    Point2f d2 = p2 - o2;

	    float cross = d1.x*d2.y - d1.y*d2.x;
	    if (abs(cross) < /*EPS*/1e-8)
	       continue;

        double temp1 = (x.x * d2.y - x.y * d2.x)/cross;
	    Point2f r1 = o1 + d1 * temp1;
		// 2nd point

		 o1=Point2f(float(boardEdges[0][0]), float(boardEdges[0][1]));
		 p1=Point2f(float(boardEdges[0][2]), float(boardEdges[0][3]));
		 o2=Point2f(float(boardEdges[3][0]), float(boardEdges[3][1]));
		 p2=Point2f(float(boardEdges[3][2]), float(boardEdges[3][3]));
		 x = o2 - o1;
	     d1 = p1 - o1;
	     d2 = p2 - o2;

	     cross = d1.x*d2.y - d1.y*d2.x;
	    if (abs(cross) < /*EPS*/1e-8)
	       continue;

         temp1 = (x.x * d2.y - x.y * d2.x)/cross;
	    Point2f r2 = o1 + d1 * temp1;
		//
		// 3rd point

		 o1=Point2f(float(boardEdges[1][0]), float(boardEdges[1][1]));
		 p1=Point2f(float(boardEdges[1][2]), float(boardEdges[1][3]));
		 o2=Point2f(float(boardEdges[2][0]), float(boardEdges[2][1]));
		 p2=Point2f(float(boardEdges[2][2]), float(boardEdges[2][3]));
		 x = o2 - o1;
	     d1 = p1 - o1;
	     d2 = p2 - o2;

	     cross = d1.x*d2.y - d1.y*d2.x;
	    if (abs(cross) < /*EPS*/1e-8)
	       continue;

         temp1 = (x.x * d2.y - x.y * d2.x)/cross;
	    Point2f r3 = o1 + d1 * temp1;
		//
		// 4th point

		 o1=Point2f(float(boardEdges[2][0]), float(boardEdges[2][1]));
		 p1=Point2f(float(boardEdges[2][2]), float(boardEdges[2][3]));
		 o2=Point2f(float(boardEdges[3][0]), float(boardEdges[3][1]));
		 p2=Point2f(float(boardEdges[3][2]), float(boardEdges[3][3]));
		 x = o2 - o1;
	     d1 = p1 - o1;
	     d2 = p2 - o2;

	     cross = d1.x*d2.y - d1.y*d2.x;
	    if (abs(cross) < /*EPS*/1e-8)
	       continue;

         temp1 = (x.x * d2.y - x.y * d2.x)/cross;
	    Point2f r4 = o1 + d1 * temp1;
		//
		board2DCorners.push_back(r1);
		board2DCorners.push_back(r2);
		board2DCorners.push_back(r3);
		board2DCorners.push_back(r4);
	//	cout<<board2DCorners.size();
	//	cout<<board2DCorners<<endl;
	//	getchar();
		//
		if (board2DCorners.size() < 4) continue;

		// estimate the outer corners' depth
		vector<short> boardCornersDepth;
		for (int j=0;j<4;j++){
			Point2f dstPoint;
			if (!findClosestPoint(segmentedDepthMat, board2DCorners[j], dstPoint)) 
				break;
			boardCornersDepth.push_back(depthMat.at<short>(floor(dstPoint.y+0.5),floor(dstPoint.x+0.5)));
			circle(depthMat1,board2DCorners[j],2,Scalar(0,255,0),2);
			circle(depthMat1,dstPoint,2,Scalar(255,0,0),2);
		}
		if (board2DCorners.size() < 4) continue;
		cv::imshow( "Depth image", depthMat1 );
		cv::waitKey( 20 );

		// a good frame -> save all useful information
		goodFrames.push_back(frame);
		incorner2DPoints[RGB_SENSOR].push_back(rgbPointBuf);
		incorner2DPoints[EXT_SENSOR].push_back(extPointBuf);
		boardCorner2DPoints.push_back(board2DCorners);
		allBoardCornerDepth.push_back(boardCornersDepth);
		rvecs[RGB_SENSOR].push_back(rVec);
		tvecs[RGB_SENSOR].push_back(tVec);

		// combine the processed images
		Mat tmpOut;
		hconcat(rgbMat,extResized,tmpOut);
		//
		Mat outView;
		hconcat(depthMat1,tmpOut,outView);
		cv::namedWindow( "Kinect Calibration", 1 );
		cv::imshow( "Kinect Calibration", outView );
		cv::waitKey( 100 );

		if (goodFrames.size() >= nFrame2Use){
			runCalibration(inCorners3D, imageSize[EXT_SENSOR],  cameraMatrix[EXT_SENSOR], distCoeffs, incorner2DPoints[EXT_SENSOR],rvecs[EXT_SENSOR],tvecs[EXT_SENSOR]);
			extCalibrated = true;
			break;
		}
	}

	if (!extCalibrated) {
		runCalibration(inCorners3D, imageSize[EXT_SENSOR],  cameraMatrix[EXT_SENSOR], distCoeffs, incorner2DPoints[EXT_SENSOR],rvecs[EXT_SENSOR],tvecs[EXT_SENSOR]);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////             STEP 2: Calibrate the depth camera             ////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////
	double cx = imageSize[DEPTH_SENSOR].width/2;		// the depth sensor's principal point
	double cy = imageSize[DEPTH_SENSOR].height/2;
	
	// Compute translation t1 between the RGB & depth sensor
	Mat t1 = Mat::zeros(3,1,CV_32F);
	for (int i=0;i<goodFrames.size();i++){
		Mat matR;
		// TODO I.3: Compute translation t1 between the RGB & depth sensor
		//           (see project decription for more details)
		// Inputs:
		//	  - Rotation vector for RGB image (rvecs[RGB_SENSOR][i])
		//	  - Translation vector for RGB image (tvecs[RGB_SENSOR][i])
		//    - 3-D coordinates of the board corners in {B} (matBoardCorners3D)
		//    - Detected 2-D board corners in the depth image (boardCorner2DPoints)
		//    - Board corners' depth in {D} (allBoardCornerDepth)
		// Task:
		//	  update t1 according to equation (6)
		//
		// fill your code here
		Rodrigues(rvecs[RGB_SENSOR][i], matR);
		for (int j = 0; j < 4; ++j) {			
			float PDM[3] = { ((boardCorner2DPoints[i][j].x - cx)/ fd) * allBoardCornerDepth[i][j] , ((boardCorner2DPoints[i][j].y - cy) / fd) * allBoardCornerDepth[i][j], allBoardCornerDepth[i][j]};
			t1 += Mat(3 , 1 , CV_32F , &PDM ) - matR * matBoardCorners3D.col(j) - tvecs[RGB_SENSOR][i];
		}
	}
	t1 /= (4*goodFrames.size());
	cout<<t1<<endl;
	//getchar();
	////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////             STEP 3: Calibrate the external camera          ////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////
	Mat r2;
	Mat t2;
	vector<Point3f> point3D;
	vector<Point2f> point2D;
	for (int i=0;i<goodFrames.size();i++){
		// TODO II.1.3: collect the inner corners' 3-D coordinates in {D} and their projection in {E}
		//              (see project decription for more details)
		// Inputs:
		//	  - Rotation vector for RGB image (rvecs[RGB_SENSOR][i])
		//	  - Translation vector for RGB image (tvecs[RGB_SENSOR][i])
		//	  - Translation vector between the RGB & the depth sensor (t1)
		//    - Detected inner corners in the external image (incorner2DPoints[EXT_SENSOR])
		// Task:
		//	  update point3D & point2D
		//
		Mat matRext;
		Mat tempmat=Mat::zeros(3,1,CV_64F);
		
		for (vector<Point2f>::iterator iter = incorner2DPoints[EXT_SENSOR][i].begin(); iter != incorner2DPoints[EXT_SENSOR][i].end(); ++iter) {
			point2D.push_back(*iter);
		}

		
		Rodrigues(rvecs[RGB_SENSOR][i], matRext);

		for (int j = 0; j < matInCorners3D.cols; ++j) {

			tempmat = matRext * matInCorners3D.col(j);
			tempmat+= tvecs[RGB_SENSOR][i];
			tempmat+= t1;
		//	cout<<tempmat<<endl;
		
			point3D.push_back(Point3f(tempmat.at<float>(0,0), tempmat.at<float>(1,0), tempmat.at<float>(2,0)));
		 	
		}	
//point3D=rvecs[RGB_SENSOR][i]*point2D.+tvecs[RGB_SENSOR][i]+t1;
	//	t1 += Mat(3 , 1 , CV_32F , &PDM ) - matR * matBoardCorners3D.col(j) - tvecs[RGB_SENSOR][i];
		
		//getchar();
	//	point2D.swap(incorner2DPoints[EXT_SENSOR]);
	//	point2D.push_back(incorner2DPoints[EXT_SENSOR]);

		//
	}
	// TODO II.1.4: use solvePnPRansac to compute r2 & t2
	//
	// fill your code here
	solvePnPRansac(point3D, point2D, cameraMatrix[EXT_SENSOR], distCoeffs, r2, t2);
	r2.convertTo(r2, CV_32F);
	t2.convertTo(t2, CV_32F);
	cout<<r2<<endl;
	cout<<t2<<endl;
	
	//


	////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////         STEP 4: Paint depth images by RGB & external images        ////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////
	for (int i=0;i<goodFrames.size();i++){
		int frame = goodFrames.at(i);
		Mat depthMat, rgbMat0, extMat0, rgbMat, extMat;

		if (!loadImages(src, frame, depthMat, rgbMat0, extMat0)){
			continue;
		}

		// comput 3D coordinates of every pixel in the depth map in {D}
		vector<Point3f> _3DPoints;
		vector<int> pSet, qSet;							// saved row & col indices of _3DPoints
		for (int p=0;p<depthMat.rows;p++){				// row index
			for (int q=0;q<depthMat.cols;q++){			// col index
				if (depthMat.at<short>(p,q) != 0){
					float X, Y, Z;
					// TODO I.4.a: compute 3D coordinate (X,Y,Z) of each pixel in the depth image according to {D}
					//
					// fill your code here
					
					X = (q - cx);
					Y = (p - cy);
					Z = depthMat.at<short>(p,q);
					X *=Z/fd;
					Y *=Z/fd;
					//
					_3DPoints.push_back(Point3f(X,Y,Z));
					pSet.push_back(p);
					qSet.push_back(q);
				}
			}
		}

		////////   From the RGB image
		printf("Projecting the RGB image...\n");
		vector<Point2f> projectedPoints;
		Mat outTest = Mat::zeros(rgbMat0.size(),rgbMat0.type());
		int pointCount = 0;
		// TODO I.4.b: 
		//     - Project these 3D points (_3DPoints) onto the color image using transformation [I, -t1]
		//	   - If the projected point is inside the RGB image, pick its color & paint the corresponding point 
		//       (see pSet & qSet) in the output image (outTest)
		//       You may want to use avSubPixelValue8U3 to get a smoother result (optional)
		//	   - Count the number of the ploted points
		//
		// fill your code here
		//
		Mat zeroCoeffs(1,4,CV_32F);
		zeroCoeffs = 0;
		
		cv::projectPoints(_3DPoints, Mat::zeros(3,1,CV_32F), (-1*t1), cameraMatrix[RGB_SENSOR],zeroCoeffs  , projectedPoints);

		for(int i=0; i<projectedPoints.size();i++)
		{
			int utemp=pSet[i];
			int vtemp=qSet[i];
			if(projectedPoints[i].y>0 && projectedPoints[i].y<rgbMat0.rows && projectedPoints[i].x>0 && projectedPoints[i].x<rgbMat0.cols)
			{
			outTest.at<Vec3b>(utemp,vtemp)=rgbMat0.at<Vec3b>(projectedPoints[i].y,projectedPoints[i].x);
			pointCount++;
			}
		
		}
		// save the output image
		sprintf(fpath,"%s/rgb_%04d.png",dst,frame);
		imwrite(fpath, outTest);
		// save the output point cloud
		sprintf(fpath,"%s/rgb_%04d.ply",dst,frame);
		writePLY(pointCount,fd,outTest,depthMat,fpath);

		
		////////   From the external image
		printf("Projecting the external image...\n");

		vector<Point3f> projectedPoints2;
		Mat outTest2 = Mat::zeros(rgbMat0.size(),rgbMat0.type());
		Mat extDepth = Mat::ones(extMat0.rows/4,extMat0.cols/4,CV_32F) * 1E+10;
		int pointCount2 = 0;
		Mat outDepth = Mat::zeros(depthMat.size(),depthMat.type());		// save depth value of ploted points
																		// used when writing the point cloud
		// TODO II.2: 
		//     - Project these 3D points (_3DPoints) onto the external image using transformation [r2, t2]
		//	   - Update the Z-buffer (extDepth).
		//     - Project these 3D points (_3DPoints) again onto the external image. 
		//		 If a 3D point has depth < 120% the stored depth in Z-buffer:
		//			+ Pick color & paint the corresponding point (see pSet & qSet) in the output image (outTest2)
		//			  You may want to use avSubPixelValue8U3 to get a smoother result (optional)
		//			+ Set its depth in outDepth
		//			+ Count the number of the ploted points
		//       Otherwise, skip the point
		//
		// fill your code here 
		vector<Point3f> newPoint;
		vector<bool> yes;
		
		Mat matRRR;
		Rodrigues(r2, matRRR);
		Point2f pp= Point2f(cameraMatrix[EXT_SENSOR].at<double>(0, 0),cameraMatrix[EXT_SENSOR].at<double>(1, 1));
		Point2f qq= Point2f(cameraMatrix[EXT_SENSOR].at<double>(0, 2),cameraMatrix[EXT_SENSOR].at<double>(1, 2));
		

		for (int i = 0; i < _3DPoints.size(); ++i) {
			Mat tempMatrix = matRRR * Mat(3, 1, CV_32F, &_3DPoints[i]) + t2;
			newPoint.push_back(Point3f(tempMatrix.at<float>(0,0)*pp.x/tempMatrix.at<float>(2,0) + qq.x,tempMatrix.at<float>(1,0)* pp.y/tempMatrix.at<float>(2,0) + qq.y,tempMatrix.at<float>(2,0)));
		
			if (newPoint[i].x > 0 && newPoint[i].x < extMat0.cols - 1.5 && newPoint[i].y > 0 && newPoint[i].y < extMat0.rows - 1) {
				extDepth.at<float>(floor((newPoint[i].y + 0.5) / 4), floor((newPoint[i].x + 0.5) / 4)) = min(extDepth.at<float>(floor((newPoint[i].y + 0.5) / 4), floor((newPoint[i].x + 0.5) / 4)), tempMatrix.at<float>(2,0));
				yes.push_back(true);
			}
			else
				yes.push_back(false);
		}
		
		for (int i = 0; i < _3DPoints.size(); ++i) {
			if (yes[i])
			{
				if (newPoint[i].z <  extDepth.at<float>((int)((newPoint[i].y + 0.5) / 4), (int)((newPoint[i].x + 0.5) / 4)) * 1.2) {
					pointCount2++;
					Vec3b val = avSubPixelValue8U3(Point2f(newPoint[i].x, newPoint[i].y), extMat0);
					outDepth.at<short>(pSet[i], qSet[i]) =newPoint[i].z;
					outTest2.at<Vec3b>(pSet[i], qSet[i]) = val;
				}
			}
		}
		
		// save the output image
		sprintf(fpath,"%s/ext_%04d.png",dst,frame);
		imwrite(fpath,outTest2);
		// save the output point cloud
		sprintf(fpath,"%s/ext_%04d.ply",dst,frame);
		writePLY(pointCount2,fd,outTest2,outDepth,fpath);
		
		// output the depth map
		Mat depthMat0 = depthMat / 4;
		depthMat0.convertTo(depthMat0, CV_8U);

		Mat depthMat1(depthMat.rows,depthMat.cols,CV_8UC3);
		insertChannel(depthMat0,depthMat1,0);
		insertChannel(depthMat0,depthMat1,1);
		insertChannel(depthMat0,depthMat1,2);

		// combine output & display
		Mat tmpOut,outfinal;
		hconcat(outTest,outTest2,tmpOut);
		hconcat(depthMat1,tmpOut,outfinal);

		cv::imshow( "Kinect Calibration", outfinal );
		cv::waitKey( 50 );
	}
	return 0;
}
