// test.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.

//link to the filter_site: https://github.com/atilimcetin/guided-filter
#include "stdafx.h"
#include <iostream>
#include <stdio.h>
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

		Mat tempLeft(imgLeft.rows, imgLeft.cols, CV_32FC1);		//matrices for storing cost volume temporarily
		Mat tempRight(imgRight.rows, imgRight.cols, CV_32FC1);

		int costLeft = 0;
		int costRight = 0;

		int normLeft = 0;
		int normRight = 0;

		int borderSize = windowSize/2;
		int windowArea = windowSize*windowSize;

		for(int i=0; i<=maxDisp; i++){		// for-loop for disparity values

			for (int x=0; x<imgLeft.cols; x++){		//for-loops for going through picture per pixel
				for (int y=0; y<imgLeft.rows; y++){

					int xStart = x-borderSize-i;
					int xEnd = x+borderSize+1-i;
					int yStart = y-borderSize;
					int yEnd = y+borderSize+1;

					for (int a=xStart; a<xEnd; a++){		//for-loops for scanning window
						for (int  b=yStart; b<yEnd; b++){


							if((a<0)||(b<0)||(a+i>imgLeft.cols-1)||(b>imgLeft.rows-1))		//setting costVolume very high for out-of-image-windows
								costLeft+=255; 

							else{				//calculating costVolums
								costLeft+=abs(imgLeft.at<uchar>(b,a+i) - imgRight.at<uchar>(b,a));
								normLeft++;
							}
							/*		right window from the right side
							if((imgRight.cols-1-a+i<0)||(b<0)||(imgRight.cols-1-a+2*i>imgLeft.cols-1)||(b>imgLeft.rows-1))		//setting costVolume very high for out-of-image-windows
								costRight+=255;

							else{				//calculating costVolums
								costRight+=abs(imgRight.at<uchar>(b,imgRight.cols-1-a+i) - imgLeft.at<uchar>(b,imgRight.cols-1-a+2*i));
								normRight++;
							}*/

							if((a+i<0)||(b<0)||(a+2*i>imgLeft.cols-1)||(b>imgLeft.rows-1))		//setting costVolume very high for out-of-image-windows
								costRight+=255;

							else{				//calculating costVolums
								costRight+=abs(imgRight.at<uchar>(b,a+i) - imgLeft.at<uchar>(b,a+2*i));
								normRight++;
							}
						}		
					}

					if (normRight==0)
						normRight=1;
					if (normLeft==0)
						normLeft=1;

					costLeft = costLeft/normLeft;
					costRight = costRight/normRight;

					tempLeft.at<float>(y,x) = costLeft;			//write cost volume in temporary matrix and reset it
					tempRight.at<float>(y,x) = costRight;
					costLeft = 0;
					costRight = 0;
					normLeft=0;
					normRight=0;
				}
			}
			costVolumeLeft.push_back(tempLeft);			//writing cost Volume matrix in vector and reset it
			costVolumeRight.push_back(tempRight);

			Mat temp2(imgLeft.rows, imgLeft.cols, CV_32FC1);
			Mat temp3(imgRight.rows, imgRight.cols, CV_32FC1);
			tempLeft = temp2;
			tempRight = temp3;
		}
	/*
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
				else{
					costLeft = 255;
					costRight = 255;
				}
					
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
	*/
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
		costVolumeRight.at(i) = guidedFilter(imgRight, costVolumeRight.at(i), r/4, eps);
	}
}

void refineDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, int scaleDispFactor){

	int dlr = 1;
	int** tempArr = new int*[dispLeft.rows];
	int** tempArrRight = new int*[dispLeft.rows];
	for(int i=0; i< dispLeft.rows; ++i){
		tempArr[i] = new int[dispLeft.cols];
		tempArrRight[i] = new int[dispLeft.cols];
	}

	for (int x=0; x<dispLeft.cols; ++x){		//for-loops for going through picture per pixel
		for (int y=0; y<dispLeft.rows; ++y){
			if(x>=scaleDispFactor){
				if(abs(dispLeft.at<uchar>(y,x)-dispRight.at<uchar>(y,x-scaleDispFactor))>dlr){ //consistency check
					tempArr[y][x]=1;  //label pixel as inconsistent
				}else{
					tempArr[y][x]=0;
				}
			}else{
				tempArr[y][x]=1;  //the pixels at the left are set as inconsistent
			}

			if(x+scaleDispFactor < dispRight.cols){
				if(abs(dispRight.at<uchar>(y,x)-dispLeft.at<uchar>(y,x+scaleDispFactor))>dlr){ //consistency check
					tempArrRight[y][x]=1;  //label pixel as inconsistent
				}else{
					tempArrRight[y][x]=0;
				}
			}else{
				tempArrRight[y][x]=1; //the pixels at the right are set as inconsistent
			}
		}
	}
	for (int x=0; x<dispLeft.cols; ++x){		//for-loops for going through picture per pixel
		for (int y=0; y<dispLeft.rows; ++y){
			if(tempArr[y][x]==1){
				int tempLeft=256;		//left Border
				int tempRight=257;		//right Border
				for(int i=x; i>0; --i){
					if(tempArr[y][i]==0){  //find next non-labeled pixel left
						tempLeft = i;
						i=0; //end for-loop
					}
				}
				for(int i=x; i<dispLeft.cols; ++i){
					if(tempArr[y][i]==0){ //find next non-labelde pixel right
						tempRight = i;
						i=dispLeft.cols; //end for-loop
					}
				}
				if(tempLeft<tempRight){
					if(tempLeft != 256)
					dispLeft.at<uchar>(y,x)=dispLeft.at<uchar>(y,tempLeft); //set nearest consistent pixel
				}else{
					if(tempRight!= 257)
					dispLeft.at<uchar>(y,x)=dispLeft.at<uchar>(y,tempRight); //set nearest consistent pixel
				}
			}

			if(tempArrRight[y][x]==1){
				int tempLeft=256;		//left Border
				int tempRight=257;		//right Border
				for(int i=x; i>0; --i){
					if(tempArrRight[y][i]==0){  //find next non-labeled pixel left
						tempLeft = i;
						i=0; //end for-loop
					}
				}
				for(int i=x; i<dispLeft.cols; ++i){
					if(tempArrRight[y][i]==0){ //find next non-labelde pixel right
						tempRight = i;
						i=dispLeft.cols; //end for-loop
					}
				}
				if(tempLeft<tempRight){
					if(tempLeft != 256)
					dispRight.at<uchar>(y,x)=dispRight.at<uchar>(y,tempLeft);	//set nearest consistent pixel
				}else{
					if(tempRight!= 257)
					dispRight.at<uchar>(y,x)=dispRight.at<uchar>(y,tempRight);	//set nearest consistent pixel
				}
			}
		}
	}
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

	int r = 10;
	double eps = 0.0000001;

	convertToGrayscale(imgLeft, imgGrayLeft);
	convertToGrayscale(imgRight, imgGrayRight);

	computeCostVolume(imgGrayLeft, imgGrayRight, costVolumeLeft, costVolumeRight, windowSize, maxDisp);

	aggregateCostVolume(imgLeft, imgGrayRight, costVolumeLeft, costVolumeRight, r, eps);

	selectDisparity(dispLeft, dispRight, costVolumeLeft, costVolumeRight, scaleDispFactor);

	refineDisparity(dispLeft, dispRight, scaleDispFactor);

	Mat dispLeftMedian;
	medianBlur(dispRight,dispLeftMedian,3);

	imshow("dispLeftAfter",dispLeftMedian);
	
	waitKey(0);
	return 0;
}

