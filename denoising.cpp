

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "src/OnlineDictionaryLearning.h"
#include <iostream>

cv::Mat_<Real> generate2DPatches(cv::Mat_<Real> img, int patchHeight, int patchWidth);
cv::Mat_<Real> reconstructImgFromPatches(cv::Mat_<Real> data, int patchHeight, int patchWidth, int imgHeight, int imgWidth);
void visualizeDictionary(Real *Dt, cv::Mat, int patchHeight, int patchWidth, int nComponents, int patchesAlongX);

int main(int argc, char** argv){

	if(argc != 3){
		std::cout << "Usage: ./applicationName originalImage.png distortedImage.png!" << std::endl;
		return -1;
	}

	const char* originalImagePath = argv[1];
	const char* distortedImagePath = argv[2];

	// constants used for computations
	int patchHeight = 10;
	int patchWidth = 10;
	int nComponents = 250;
	// lengthOfComponent is lengthOfPatch
	int lengthOfComponent = patchHeight*patchWidth;
	int nIterations = 4000;
	Real regularizationParameter = 1.0;
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

	originalImageGray.convertTo(originalImageGrayFloat, CV_64F, 1.0/255.0);
	distortedImageGray.convertTo(distortedImageGrayFloat, CV_64F, 1.0/255.0);




	// extract patches from original image
	std::cout<<"Extracting patches from original image... "<<std::endl;
	cv::Mat_<Real> originalPatches = generate2DPatches(originalImageGrayFloat, patchHeight, patchWidth);
	std::cout<<"Extracting patches from original image completed. "<<std::endl;

	/*
	//this part checks that the methods I implemented: generate2DPatches and reconstructImgFromPatches
	cv::Mat rec;
	rec = reconstructImgFromPatches(originalPatches, patchHeight, patchWidth, originalImageGrayFloat.rows, originalImageGrayFloat.cols);
	cv::namedWindow("Visualize", CV_WINDOW_AUTOSIZE );
	cv::imshow("Visualize",rec);
	cv::waitKey(0);
	*/

	// normalize along columns: with mean 0 and std. 1 for each column.
	cv::Mat originalPatchesMean;
	cv::Mat originalPatchesStd;

	for(int i = 0; i < originalPatches.cols; i ++){
		cv::Mat originalPatchesColumnwiseMean;
		cv::Mat originalPatchesColumnwiseStd;

		cv::Mat col = originalPatches.col(i);
		cv::meanStdDev(col, originalPatchesColumnwiseMean, originalPatchesColumnwiseStd);

		originalPatches.col(i) -= (Real)originalPatchesColumnwiseMean.at<Real>(0);
		originalPatches.col(i) /= (Real)originalPatchesColumnwiseStd.at<Real>(0);

		originalPatchesMean.push_back((Real)originalPatchesColumnwiseMean.at<Real>(0));
		originalPatchesStd.push_back((Real)originalPatchesColumnwiseStd.at<Real>(0));
	}
	std::cout<<originalPatchesStd<<std::endl;
	std::cout<<originalPatchesMean<<std::endl;
//	cv::namedWindow("Original Imag1e", CV_WINDOW_AUTOSIZE );
//	cv::imshow("Original Imag1e", originalImageGrayFloat);
//	cv::waitKey(0);


	// dictionary learning part
	std::cout<<"Learning dictionary from original image..."<<std::endl;
	Timer timer(END_ITR);
	DictionaryLearning learnDict(regularizationParameter, lengthOfComponent, nComponents, timer);

	std::cout<<originalPatches.row(1);
	std::cout<<originalPatches.rows<<"  "<<originalPatches.cols<<std::endl;
	for(int i = 0; i < nIterations; i++){
		learnDict.iterate((Real*)(originalPatches.row(i%originalPatches.rows)).data);
		if (i%200 == 0){
			std::cout<<"Iteration: "<<i<<std::endl;
		}
	}

	std::cout<<"Learned Dictionary"<<std::endl;
	for(int c = 0; c < nComponents; c++){
		for(int d = 0; d < patchHeight*patchWidth; d++){
			std::cout<<learnDict.Dt[c*patchHeight*patchWidth + d]<<" ";
			// visualize dictionary
		}
		std::cout<<std::endl;
	}

	std::cout<<"Learning dictionary from original image completed."<<std::endl;	

	// generate patches from original image, here we need to store mean and std, so it is slightly different than above
	std::cout<<"Extracting patches from noisy image... "<<std::endl;
	cv::Mat_<Real> distortedPatches = generate2DPatches(distortedImageGrayFloat, patchHeight, patchWidth);
	std::cout<<"Extracting patches from noisy image completed."<<std::endl;

	std::cout<<"Normalizing noisy data image..."<<std::endl;
	cv::Mat distortedPatchesMean;
	cv::Mat distortedPatchesStd;

	for(int i = 0; i < distortedPatches.cols; i++){
		cv::Mat distortedPatchesColumnwiseMean;
		cv::Mat distortedPatchesColumnwiseStd;
		cv::Mat col;
		col = distortedPatches.col(i);
		cv::meanStdDev(col, distortedPatchesColumnwiseMean, distortedPatchesColumnwiseStd);
		distortedPatchesMean.push_back(distortedPatchesColumnwiseMean.at<Real>(0));
		distortedPatchesStd.push_back(distortedPatchesColumnwiseStd.at<Real>(0));
	}

	// substract mean
	for(int i = 0; i < distortedPatches.cols; i++){
		distortedPatches.col(i) -= distortedPatchesMean.at<Real>(i);
	}
	std::cout<<"Normalizing noisy data image completed."<<std::endl;


	int patchesAlongX = 10;
	//std::cout<<distortedPatchesMean<<std::endl;
	visualizeDictionary(learnDict.Dt, distortedPatchesMean, patchHeight, patchWidth, nComponents, patchesAlongX);


	// reconstruct each patch of damaged image
	std::cout<<"Reconstructing noisy image patches using dictionary learned from original image..."<<std::endl;
	cv::Mat_<Real> reconstructedPatches(distortedPatches.rows, distortedPatches.cols);
	std::cout<<"Number of vectors to be reconstructed: "<<reconstructedPatches.rows<<std::endl;
	for(int i = 0; i < reconstructedPatches.rows; i++){
		//std::cout<<i<<std::endl;
		if (i%10000 == 0){
			std::cout<<"Reconstructing Patch: "<<i<<std::endl;
		}
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
	timer.print(1000);
	cv::namedWindow("Original Image", CV_WINDOW_NORMAL );
	cv::namedWindow("Distorted Image", CV_WINDOW_NORMAL );
	cv::namedWindow("Reconstructed Image", CV_WINDOW_NORMAL );

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
		cv::Mat row_;
		(data.row(i)).copyTo(row_);
		row_.reshape(1,patchHeight);
		int idX = i % nPatchesAlongHorizontal;
		int idY = i / nPatchesAlongHorizontal;
		row_=row_.reshape(1, patchHeight);
		cv::Mat roi = (img(cv::Rect_<Real>(idX, idY, patchWidth, patchHeight))).clone();
		roi += row_;
		roi.copyTo(img(cv::Rect_<Real>(idX, idY, patchWidth, patchHeight)));
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

void visualizeDictionary(Real *Dt, cv::Mat mean, int patchHeight, int patchWidth, int nComponents, int patchesAlongX){
	cv::Mat img(nComponents/patchesAlongX*patchHeight, patchesAlongX*patchWidth, CV_64FC1);
	cv::Mat row(1, patchWidth*patchHeight, CV_64FC1);
	
	for (int i = 0; i < nComponents; i ++){
		cv::Mat r = cv::Mat(patchHeight, patchWidth, CV_64FC1, &Dt[i*patchWidth*patchHeight]);
		//std::cout<<r<<std::endl;
		cv::Mat temp = mean.clone();
		
		temp=temp.reshape(1,patchHeight);
		r += temp;
		//std::cout<<r<<std::endl;
		//std::cout<<i<<"  asdasd "<<std::endl;
		//mean.addTo(r);
		//cv::addWeighted(r.reshape(0,1), 1., mean, 1., 0., r);
		//row = (r.reshape(1,patchHeight)).clone();
		
		//row.reshape(1, patchHeight);
		
		r.copyTo(img(cv::Rect_<Real>((i%patchesAlongX)*patchWidth, (i/patchesAlongX)*patchHeight, patchWidth, patchHeight)));
	}

	cv::namedWindow("LearnedPatches", CV_WINDOW_NORMAL);
	cv::imshow("LearnedPatches", img);
	cv::waitKey(0);
}



cv::Mat_<Real> generate2DPatches(cv::Mat_<Real> img, int patchHeight, int patchWidth){
	int nPatchesAlongHorizontal = img.cols - patchWidth + 1;
	int nPatchesAlongVertical = img.rows - patchHeight + 1;
	cv::Mat_<Real> data = cv::Mat_<Real>::zeros(nPatchesAlongVertical*nPatchesAlongVertical, patchHeight*patchWidth);
	for(int i = 0; i < nPatchesAlongVertical; i++){
		for(int j = 0; j < nPatchesAlongHorizontal; j++){
			//std::cout<<"patches:"<<i<<" "<<j<<std::endl;
			cv::Mat_<Real> tmp = img(cv::Rect_<Real>(j,i, patchWidth, patchHeight));
			// flatten patch to a single row
			if(!tmp.isContinuous()){
				tmp = tmp.clone();
			}
			tmp=tmp.reshape(0, 1);
			tmp.copyTo(data.row(i*nPatchesAlongHorizontal + j));
		}

	}
	return data;
}
