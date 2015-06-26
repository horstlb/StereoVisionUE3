// test.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.

//link to the filter_site: https://github.com/atilimcetin/guided-filter
#include "stdafx.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


void convertToGrayscale(const Mat &img, Mat &imgGray){
	
	for (int x=0; x<img.cols; ++x){		//for-loops for going through picture per pixel
		for (int y=0; y<img.rows; ++y){
			
			Vec3b intensity = img.at<Vec3b>(y, x);		//calculating luminance value for each pixel
			uchar blue = intensity.val[0];				//and pass it to the grayscale image
			uchar green = intensity.val[1];
			uchar red = intensity.val[2];	
			uchar grey = 0.21*red+0.72*green+0.07*blue;
			
			imgGray.at<uchar>(y, x) = grey;			
		}
	}	
}

void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, vector<Mat> &costVolumeLeft, 
	vector<Mat> &costVolumeRight, int windowSize, int maxDisp){
		
	int borderSize = windowSize/2;
	float costLeft = 0;
	float costRight = 0;
	float leftRight = 0;
	float leftLeft = 0;
	float rightLeft = 0;
	float rightRight = 0;
	
	Mat tempLeft(imgLeft.rows, imgLeft.cols, CV_32FC1);		//matrices for storing cost volume temporarily
	Mat tempRight(imgRight.rows, imgRight.cols, CV_32FC1);
	
	for (int d=0; d<maxDisp;d++){				//loops for disparity, rows and cols
		for (int i = borderSize; i<imgLeft.rows-borderSize; i++){
			for (int j = borderSize; j<imgLeft.cols-borderSize; j++){
				int startY = i - borderSize;		//defining scanning window
				int endY = i + borderSize + 1;
				int startX = j - borderSize-d;
				int endX = j + borderSize + 1-d;
				
				if (startX==0){		//if start of row
					for (int k=startY;k<endY; k++){
						for (int l=startX; l<endX; l++){
							costLeft+=abs(imgLeft.at<uchar>(k,l+d)-imgRight.at<uchar>(k,l));
							costRight+=abs(imgRight.at<uchar>(k,l+d)-imgLeft.at<uchar>(k,l+2*d));
						}
					}
				}
				
				else if (startX>0){		//sliding window approach
					leftLeft=0;
					leftRight=0;
					rightLeft=0;
					rightRight=0;
					for(int k = startY; k < endY; k++){
						leftLeft+=abs(imgLeft.at<uchar>(k,startX-1+d)-imgRight.at<uchar>(k,startX-1));		//left side
						leftRight+=abs(imgLeft.at<uchar>(k,endX-1+d)-imgRight.at<uchar>(k,endX-1));			//right side
						if (endX+2*d<imgLeft.cols){
							rightLeft+=abs(imgRight.at<uchar>(k,startX-1+d)-imgLeft.at<uchar>(k,startX-1+2*d));
							rightRight+=abs(imgRight.at<uchar>(k,endX-1+d)-imgLeft.at<uchar>(k,endX-1+2*d));
						}
					}
					costLeft = tempLeft.at<float>(i,j-1)-leftLeft+leftRight;		//compute sliding window
					costRight = tempRight.at<float>(i,j-1)-rightLeft+rightRight;
				}
				else
					costLeft = 255;

				tempLeft.at<float>(i,j) = costLeft;			//storing costVolume
				tempRight.at<float>(i,j) = costRight;
				costLeft=0;
				costRight=0;
			}
		}
		costVolumeLeft.push_back(tempLeft);			//writing cost Volume matrix in vector and reset it
		costVolumeRight.push_back(tempRight);
		Mat temp2(imgLeft.rows, imgLeft.cols, CV_32FC1);	
		Mat temp3(imgLeft.rows, imgLeft.cols, CV_32FC1);	
		tempLeft=temp2;
		tempRight=temp3;
	}
	
	

}

void selectDisparity(Mat &dispLeft, Mat &dispRight, vector<Mat> &costVolumeLeft, 
	vector<Mat> &costVolumeRight, int scaleDispFactor){
	
	float displevelLeft = 255;
	float displevelRight = 255;

	int disperityLeft = 0;
	int disperityRight = 0;

	float costVolLeft = 0;
	float costVolRight = 0;

	for (int x=0; x<dispLeft.cols; ++x){		//for-loops for going through picture per pixel
		for (int y=0; y<dispLeft.rows; ++y){

			for(int i=0; i<costVolumeLeft.size(); i++){			//for-loops for different disparity values

				costVolLeft = costVolumeLeft.at(i).at<float>(y,x);
				costVolRight = costVolumeRight.at(i).at<float>(y,x);

				if (costVolLeft<displevelLeft){				//check if current cost volume is lower then previously lowest
				  displevelLeft = costVolumeLeft.at(i).at<float>(y,x);			//if so, set it as new lower boundary
				  disperityLeft = i;
				}  

				if (costVolRight<displevelRight){
					displevelRight = costVolumeRight.at(i).at<float>(y,x);
					disperityRight = i;
				}
			}

			dispLeft.at<uchar>(y,x)=disperityLeft*scaleDispFactor;			//set pixel in desparity map
			dispRight.at<uchar>(y,x)=disperityRight*scaleDispFactor;			//set pixel in desparity map
			disperityLeft = 0;
			disperityRight = 0;
			displevelLeft = 255;
			displevelRight = 255;
		}
	}
}

void aggregateCostVolume(const cv::Mat &imgLeft, const cv::Mat &imgRight, std::vector<cv::Mat>
&costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int r, double eps){
	
	eps *= 255*255;
	for (int i = 0; i<costVolumeLeft.size(); i++){
		costVolumeLeft.at(i) = guidedFilter(imgLeft, costVolumeLeft.at(i), r, eps);
		costVolumeRight.at(i) = guidedFilter(imgRight, costVolumeRight.at(i), r, eps);
	}
}

void refineDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, int scaleDispFactor){

}

int main(){
	const Mat imgLeft = imread("tsukuba_left.png", CV_LOAD_IMAGE_UNCHANGED);		//reading the two original images
	const Mat imgRight = imread("tsukuba_right.png", CV_LOAD_IMAGE_UNCHANGED);
	
	vector<Mat> costVolumeLeft;			//vectors for storing the different cost volumes for each disparity
	vector<Mat> costVolumeRight;
	
	int windowSize = 5;			//scanning window size
	int maxDisp = 15;			//definition of the maximum disparity
	int scaleDispFactor = 16;		//stretches the disparity levels for better visualization


	Mat dispLeft(imgLeft.rows, imgLeft.cols, CV_8UC1);
	Mat dispRight(imgRight.rows, imgRight.cols, CV_8UC1);

	Mat imgGrayLeft(imgLeft.rows, imgLeft.cols, CV_8UC1);			//computing the grayscale images				
	Mat imgGrayRight(imgRight.rows, imgRight.cols, CV_8UC1);

	int r = 2; // try r=2, 4, or 8
	double eps = 0.1 * 0.1; // try eps=0.1^2, 0.2^2, 0.4^2

	convertToGrayscale(imgLeft, imgGrayLeft);
	convertToGrayscale(imgRight, imgGrayRight);

	computeCostVolume(imgGrayLeft, imgGrayRight, costVolumeLeft, costVolumeRight, windowSize, maxDisp);

	aggregateCostVolume(imgLeft, imgGrayRight, costVolumeLeft, costVolumeRight, r, eps);

	selectDisparity(dispLeft, dispRight, costVolumeLeft, costVolumeRight, scaleDispFactor);

	refineDisparity(dispLeft, dispRight, scaleDispFactor);

	imshow("dispLeft",dispLeft);
	imshow("dispRight",dispRight);

	waitKey(0);
	return 0;
}

