// C++ file



/*
arg0 is iamge filename
more args can be added later
*/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

cv::Mat_<float> generate2DPatches(cv::Mat_<float> img, int patchHeight, int patchWidth);
cv::Mat_<float> reconstructImgFromPatches(cv::Mat_<float> data, int patchHeight, int patchWidth, int imgHeight, int imgWidth);

int main(int argc, char ** argv){

	// constants used for computations
	// define patchHeight and patchWidth
	int patchHeight = 5;
	int patchWidth = 5;
	int nComponents = 100;
	// lengthOfComponent is lengthOfPatch
	int lengthOfComponent = patchHeight*patchWidth;
	int nIterations = 1000;

	if(argc < 2){
		std::cout << "Usage: ./test imageToLearnDictionary imageToDenoise." << std::endl;
		return -1;
	}

	const char* originalImagePath = argv[0];
	const char* distortedImagePath = argv[1];

	// convert to gray scale
	cv::Mat originalImageGray = cv::imread(originalImagePath, cv::IMREAD_GRAYSCALE);
	cv::Mat distortedImageGray = cv::imread(distortedImagePath, cv::IMREAD_GRAYSCALE);

	/*
	// make sure number of channels is one
	cv::Mat img_gray;
	if(img.channels()!=1){
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	} else {
		img_gray = img;
	}
	*/

	cv::Mat originalImageGrayFloat;
	cv::Mat distortedImageGrayFloat;
	originalImage.convertTo(originalImageGrayFloat, cv::CV_32F);
	distortedImage.convertTo(distortedImageGrayFloat, cv::CV_32F);

	//python script does downsampling, but we do not need to. We do it manually using softwares.

	// extract patches from distortedImage
	cv::Mat_<float> originalPatches = generate2DPatches(originalImageGrayFloat, patchHeight, patchWidth);

	// normalize along columns: with mean 0 and std. 1 for each column.
	cv::Mat originalPatchesMean;
	cv::Mat originalPatchesStd;
	for(int i = 0; i < originalPatches.cols, i ++){
		cv::meanStdDev(patches.col(i), originalPatchesMean, originalPatchesStd);
		originalPatches.col(i) -= originalPatchesMean.at<float>(0);
		originalPatches.col(i) /= originalPatchesStd.at<float>(0);
	}

	// dictionary learning part
	// TODO: needs to be completed: learn a dictionary
	// TODO: dictLearningOnline(double * X, int n_features, int n_samples, int n_components, double alpha);
	cv::Mat_<float> initialDictionary(nComponents, lengthOfComponent);
	cv::randn(initialDictionary, 0.0, 1.0);
	DictionaryLearning learnDict(regularizationParameter, initialDictionary, lengthOfComponent, nComponents);
	
	for(int i = 0; i < nIterations; i++){
		learnDict.iterate((originalPatches.row(i)).data);
	}

	// generate patches from original image, here we need to store mean and std, so it is slightly different than above
	cv::Mat distortedPatches = generate2DPatches(distortedImageGrayFloat, patchHeight, patchWidth);
	cv::Mat distortedPatchesMean;
	cv::Mat distortedPatchesStd;
	for(int i = 0; i < distortedPatches.cols, i ++){
		cv::Mat distortedPatchesColumnwiseMean;
		cv::Mat distortedPatchesColumnwiseStd;
		cv::meanStdDev(patches.col(i), distortedPatchesColumnwiseMean, distortedPatchesColumnwiseStd);
		distortedPatchesMean.at<float>(i) = distortedPatchesColumnwiseMean.at<float>(0);
		distortedPatchesStd.at<float>(i) = distortedPatchesColumnwiseStd.at<float>(0);
	}

	// substract mean
	for(int i = 0; i < distortedPatches.cols; i++){
		distortedPatches.col(i) -= distortedPatchesMean.at<float>(i);
	}

	// python portions:
	//code = dico.transform(distortedPatches);
	cv::Mat code = learnDict.sparse_coding();

	cv::Mat reconstructedPatches = dot(code, learnedDictionary);
	for(int i = 0; i < reconstructedPatches.cols; i++){
		reconstructedPatches.col(i) += distortedPatchesMean.at<float>(i);
	}

	cv::Mat reconstructedImage = reconstructImgFromPatches(reconstructedPatches, patchHeight, patchWidth, distortedImageGrayFloat.rows, distortedImageGrayFloat.cols);

	// display three windows to visualize results:
	cv::namedWindow("Original Image", cv::CV_WINDOW_AUTOSIZE );
	cv::namedWindow("Distorted Image", cv::CV_WINDOW_AUTOSIZE );
	cv::namedWindow("Reconstructed Image", cv::CV_WINDOW_AUTOSIZE );

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
cv::Mat_<float> reconstructImgFromPatches(cv::Mat_<float> data, int patchHeight, int patchWidth, int imgHeight, int imgWidth){
	//int nPatches = data.rows;
	int nPatchesAlongHorizontal = imageWidth - patchWidth + 1;
	int nPatchesAlongVertical = imageHeight - patchHeight + 1;

	cv::Mat_<float> img = cv::Mat_<float>::zeros(imgHeight, imgWidth);

	// this implementation may be different from scikit-learn
	for(int i = 0; i < data.rows; i++){
		cv::Mat_<float> row_;
		(data.row(i)).copyTo(row_);
		row_.reshape(1,patchHeight);
		int idX = i % nPatchesAlongHorizontal;
		int idY = i / nPatchesAlongHorizontal;
		row_.copyTo(img(cv::Rect_<float>(idX, idY, patchWidth, patchHeight)));
	}

	// average out over patches
	for(int i =0; i < imageHeight; i++){
		for(int j = 0; j < imageWidth; j++){
			img.at<float>(i,j) /= (float)(std::min({i+1, patchHeight, imageHeight-1})* std::min({j+1, patchWidth, imageWidth-1}));
		}
	}

	return img;
}



/*
	img is grayscale (single channel) image
	patchHeight and patchWidth indicate height and width of the patch
	extracts patches from image and returns a matrix whose rows
	represent the flattened patches of the image.

*/

cv::Mat_<float> generate2DPatches(cv::Mat_<float> img, int patchHeight, int patchWidth){
	//https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/feature_extraction/image.py#L300
	int nPatchesAlongHorizontal = img.rows - patchWidth + 1;
	int nPatchesAlongVertical = img.cols - patchHeight + 1;
	cv::Mat_<float> data = cv::Mat_<float>::zeros(nPatchesAlongVertical*nPatchesAlongVertical, patchHeight*patchWidth);
	for(int i = 0; i < nPatchesAlongHorizontal; i++){
		for(int j = 0; j < nPatchesAlongVertical; j++){
			cv::Mat_<float> tmp = img(cv::Rect_<float>(nPatchesAlongHorizontal,nPatchesAlongVertical, patchWidth, patchHeight));
			// flatten patch to a single row
			tmp.reshape(1, 1);
			// copy row to data matrix
			tmp.copyTo(data.row(j*nPatchesAlongVertical + i));
		}

	}
	return data;
}
