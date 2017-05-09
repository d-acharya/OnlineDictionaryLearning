

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "src/OnlineDictionaryLearning.h"
#include <iostream>

cv::Mat_<Real> generate2DPatches(cv::Mat_<Real> img, int patchHeight, int patchWidth);
cv::Mat_<Real> reconstructImgFromPatches(cv::Mat_<Real> data, int patchHeight, int patchWidth, int imgHeight, int imgWidth);

int main(int argc, char** argv){

	if(argc != 3){
		std::cout << "Usage: ./applicationName originalImage.png distortedImage.png!" << std::endl;
		return -1;
	}

	const char* originalImagePath = argv[1];
	const char* distortedImagePath = argv[2];

	// constants used for computations
	int patchHeight = 3;
	int patchWidth = 3;
	int nComponents = 100;
	// lengthOfComponent is lengthOfPatch
	int lengthOfComponent = patchHeight*patchWidth;
	int nIterations = 2000;
	Real regularizationParameter = 0.1;
	int transformNNonZeroCoef = 5;

	// read single channel of images
	cv::Mat originalImageGray, distortedImageGray; 
	originalImageGray = cv::imread(argv[1], 0);
	distortedImageGray = cv::imread(argv[2], 0);
	if(!originalImageGray.data || !distortedImageGray.data){
		std::cout<<"Image could not be read."<<std::endl;
		return -1;	
	}

	cv::Mat_<Real> originalImageGrayFloat;
	cv::Mat_<Real> distortedImageGrayFloat;

	originalImageGray.convertTo(originalImageGrayFloat, CV_32F, 1.0/255.0);
	distortedImageGray.convertTo(distortedImageGrayFloat, CV_32F, 1.0/255.0);



	// extract patches from original image
	std::cout<<"Extracting patches from original image... "<<std::endl;
	cv::Mat_<Real> originalPatches = generate2DPatches(originalImageGrayFloat, patchHeight, patchWidth);
	std::cout<<"Extracting patches from original image completed. "<<std::endl;

	// normalize along columns: with mean 0 and std. 1 for each column.
	cv::Mat originalPatchesMean;
	cv::Mat originalPatchesStd;
	for(int i = 0; i < originalPatches.cols; i ++){
		cv::meanStdDev(originalPatches.col(i), originalPatchesMean, originalPatchesStd);
		originalPatches.col(i) -= originalPatchesMean.at<Real>(0);
		originalPatches.col(i) /= originalPatchesStd.at<Real>(0);
	}

	// dictionary learning part
	std::cout<<"Learning dictionary from original image..."<<std::endl;
	DictionaryLearning learnDict(regularizationParameter, lengthOfComponent, nComponents);
	for(int i = 0; i < nIterations; i++){
		learnDict.iterate((Real*)(originalPatches.row(i%originalPatches.rows)).data);
	}
	std::cout<<"Learning dictionary from original image completed."<<std::endl;

	// generate patches from original image, here we need to store mean and std, so it is slightly different than above
	std::cout<<"Extracting patches from noisy image... "<<std::endl;
	cv::Mat_<Real> distortedPatches = generate2DPatches(distortedImageGrayFloat, patchHeight, patchWidth);
	std::cout<<"Extracting patches from noisy image completed."<<std::endl;
	cv::Mat distortedPatchesMean;
	cv::Mat distortedPatchesStd;
	for(int i = 0; i < distortedPatches.cols; i ++){
		cv::Mat distortedPatchesColumnwiseMean;
		cv::Mat distortedPatchesColumnwiseStd;
		cv::meanStdDev(originalPatches.col(i), distortedPatchesColumnwiseMean, distortedPatchesColumnwiseStd);
		distortedPatchesMean.at<Real>(i) = distortedPatchesColumnwiseMean.at<Real>(0);
		distortedPatchesStd.at<Real>(i) = distortedPatchesColumnwiseStd.at<Real>(0);
	}

	// substract mean
	for(int i = 0; i < distortedPatches.cols; i++){
		distortedPatches.col(i) -= distortedPatchesMean.at<Real>(i);
	}


	// reconstruct each patch of damaged image
	std::cout<<"Reconstructing noisy image patches using dictionary learned from original image..."<<std::endl;
	cv::Mat_<Real> reconstructedPatches(distortedPatches.rows, distortedPatches.cols);
	for(int i = 0; i < reconstructedPatches.rows; i++){
		learnDict.recover((Real*)(distortedPatches.row(i)).data, (Real*)(reconstructedPatches.row(i)).data);
	}
	std::cout<<"Reconstructing noisy image patches using dictionary learned from original image completed."<<std::endl;

	for(int i = 0; i < reconstructedPatches.cols; i++){
		reconstructedPatches.col(i) += distortedPatchesMean.at<Real>(i);
	}
	
	std::cout<<"Constructing image from learned patches ..."<<std::endl;
	cv::Mat reconstructedImageGrayFloat = reconstructImgFromPatches(reconstructedPatches, patchHeight, patchWidth, distortedImageGrayFloat.rows, distortedImageGrayFloat.cols);
	std::cout<<"Constructing image from learned patches completed."<<std::endl;

	// display three windows to visualize results:
	cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE );
	cv::namedWindow("Distorted Image", CV_WINDOW_AUTOSIZE );
	cv::namedWindow("Reconstructed Image", CV_WINDOW_AUTOSIZE );

	cv::imshow("Original Image", originalImageGrayFloat);
	cv::imshow("Distorted Image", distortedImageGrayFloat);
	cv::imshow("Reconstructed Image", reconstructedImageGrayFloat);

	cv::waitKey(0);

	return 0;
}


/*
patchHeight : height of the patch
patchWidth : width of the patch
data : this is opencv data type and the rows represent individual patch
implementation follows following structure:
rows of Mat_<double> represent individual patches
https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/feature_extraction/image.py#L393
*/
cv::Mat_<Real> reconstructImgFromPatches(cv::Mat_<Real> data, int patchHeight, int patchWidth, int imgHeight, int imgWidth){
	//int nPatches = data.rows;
	int nPatchesAlongHorizontal = imgWidth - patchWidth + 1;
	int nPatchesAlongVertical = imgHeight - patchHeight + 1;

	cv::Mat_<Real> img = cv::Mat_<Real>::zeros(imgHeight, imgWidth);

	for(int i = 0; i < data.rows; i++){
		cv::Mat_<Real> row_;
		(data.row(i)).copyTo(row_);
		row_.reshape(1,patchHeight);
		int idX = i % nPatchesAlongHorizontal;
		int idY = i / nPatchesAlongHorizontal;
		row_.copyTo(img(cv::Rect_<Real>(idX, idY, patchWidth, patchHeight)));
	}

	// average out over patches
	for(int i =0; i < imgHeight; i++){
		for(int j = 0; j < imgWidth; j++){
			img.at<Real>(i,j) /= (Real)(std::min(i+1, std::min(patchHeight, imgHeight-1))* std::min(j+1, std::min(patchWidth, imgWidth-1)));
		}
	}

	return img;
}



/*
	img is grayscale (single channel) image
	patchHeight and patchWidth indicate height and width of the patch
	extracts patches from image and returns a matrix whose rows
	represent the flattened patches of the image.
	//https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/feature_extraction/image.py#L300

*/

cv::Mat_<Real> generate2DPatches(cv::Mat_<Real> img, int patchHeight, int patchWidth){	
	int nPatchesAlongHorizontal = img.rows - patchWidth + 1;
	int nPatchesAlongVertical = img.cols - patchHeight + 1;
	cv::Mat_<Real> data = cv::Mat_<Real>::zeros(nPatchesAlongVertical*nPatchesAlongVertical, patchHeight*patchWidth);
	for(int i = 0; i < nPatchesAlongHorizontal; i++){
		for(int j = 0; j < nPatchesAlongVertical; j++){
			std::cout<<"patches:"<<i<<" "<<j<<std::endl;
			cv::Mat_<Real> tmp = img(cv::Rect_<Real>(nPatchesAlongHorizontal,nPatchesAlongVertical, patchWidth, patchHeight));
			// flatten patch to a single row
			tmp.reshape(1, 1);
			// copy row to data matrix
			tmp.copyTo(data.row(j*nPatchesAlongVertical + i));
		}

	}
	return data;
}
