// C++ file



/*
arg0 is iamge filename
more args can be added later
*/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

cv::Mat_<float> generate2DPatches(cv::Mat_<float> img, int patchHeight, int patchWidth);
cv::Mat_<float> reconstructFromPatches(cv::Mat_<float> data, int patchHeight, int patchWidth, int imgHeight, int imgWidth);

int main(int argc, char ** argv){

	// define patchHeight and patchWidth
	int patchHeight = 5;
	int patchWidth = 5;

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
	cv::Mat_<float> patches = generate2DPatches(originalImageGrayFloat, patchHeight, patchWidth);

	// normalize : around mean with std. 1
	cv::Mat originalPatchesMean;
	cv::Mat originalPatchesStd;
	for(int i = 0; i < patches.cols, ){
		
	}

	dim = 0, op = CV_REDUCE



	/*
	// normalize
	cv::Mat mean;
	cv::Mat stdDev;
	cv::meanStdDev(img_float, mean, stdDev);
	//img_float -= mean[0];
	//img_float /=stdDev[0];
	*/


	uchar * data = img_gray.data;
	int n_rows = img_gray.rows;
	int n_cols = img_gray.cols;

	// generate patches




}


/*
patchHeight : height of the patch
patchWidth : width of the patch
data : this is opencv data type and the rows represent individual patch
implementation follows following structure:
rows of Mat_<double> represent individual patches
https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/feature_extraction/image.py#L393
*/
cv::Mat_<float> reconstructFromPatches(cv::Mat_<float> data, int patchHeight, int patchWidth, int imgHeight, int imgWidth){
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
