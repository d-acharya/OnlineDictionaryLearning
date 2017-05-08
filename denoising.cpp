// C++ file



/*
arg0 is iamge filename
more args can be added later
*/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

int main(int argc, char ** argv){
	if(argc < 2){
		std::cout << "Usage: ./test imageToLearnDictionary imageToDenoise." << std::endl;
		return -1;
	}

	const char* image_file = argv[0];

	// convert to gray scale
	cv::Mat img_gray = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
	/*
	cv::Mat img_gray;
	if(img.channels()!=1){
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	} else {
		img_gray = img;
	}
	*/
	 
	//cv::Mat 
	
	cv::Mat img_float;
	img_gray.convertTo(img_float, cv::CV_32F);

	/*
	python script does downsampling, but we do not need to.
	*/

	/*
	load distorted image
	*/

	/*
	
	*/

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
1. load distorted image
2. load undistorted image
3. extract patches already as vectors from undistorted image and store them in an array
define my own function to return array by windowing

4. normalize individual patches
use reduce operations:
http://stackoverflow.com/questions/20883235/function-in-opencv-to-find-mean-avg-over-any-one-dimension-rows-cols-simulta
Mat row_mean, col_mean;
reduce(img,row_mean, 0, CV_REDUCE_AVG);
reduce(img,col_mean, 1, CV_REDUCE_AVG);

5. 
Learn Dictionary

6. 
a. Extract Patches from noisy image
b. normalize those noise patches (subtract mean and divide by standard deviation)
has to be done manually: no default implementation

for (int r = 0; r < M.rows; ++r) {
    M.row(r) = M.row(r) - V;
}
c.

7. 
dico.transform(data)
np.dot(code,V) (....................?)
compute dotproduct
patches = np.dot(code, V)

patches += intercept
patches 

display three images side by side


*/


/*
patchHeight : height of the patch
patchWidth : width of the patch
data : this is opencv data type and the rows represent individual patch
implementation follows following structure:
rows of Mat_<double> represent individual patches
https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/feature_extraction/image.py#L393
TODO: Can we do this more efficiently? (This is similar question as posed in scikit implementation on github.)
*/

/*
data1 = > patch1
data2 = > patch2
data3 = > patch3
data3 = > patch4

img 

*/
cv::Mat_<float> reconstructFromPatches(cv::Mat_<float> data, int patchHeight, int patchWidth, int imgHeight, int imgWidth){
	//int nPatches = data.rows;
	int nPatchesAlongHorizontal = imageWidth - patchWidth + 1;
	int nPatchesAlongVertical = imageHeight - patchHeight + 1;

	cv::Mat_<float> img = cv::Mat_<float>::zeros(imgHeight, imgWidth);

	// my own construct:
	for(int i = 0; i < data.rows; i++){
		for(int j = 0; j < data.cols; j++){
			// get imgX, imgY
			int imgX = ;
			int imgY = ;
			img.at<float>(imgX, imgY) = data.at<float>(i,j);
		}
	}
	/*
	// copy patches to image
	for(int i = 0; i < nPatchesAlongHorizontal; i++ ){
		for(int j = 0; j < nPatchesAlongVertical; j++){

		}
	}
	*/
	// average out over patches
	for(int i =0; i < imageHeight; i++){
		for(int j = 0; j < imageWidth; j++){
			img.at<float>(i,j) /= // use the computation in python website;
		}
	}

	return img;
}

/*


*/

cv::Mat_<float> generate2DPatches(cv::Mat_<float> img, int patchHeight, int patchWidth){
	//https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/feature_extraction/image.py#L300
	int nPatchesAlongHorizontal = img.rows - patchWidth + 1;
	int nPatchesAlongVertical = img.cols - patchHeight + 1;
	cv::Mat_<float> data = cv::Mat_<float>::zeros(nPatchesAlongVertical*nPatchesAlongVertical, patchHeight*patchWidth);
	for(int i = 0; i < nPatchesAlongHorizontal; i++){
		for(int j = 0; j < nPatchesAlongVertical; j++){
			cv::Mat_<float> tmp = ;
			tmp.reshape(1, 1);
			//int idX = ;
			//int idY = ;
			tmp.copyTo(data.row(j*nPatchesAlongVertical + i));
			data.at<float>(rows,cols) = img.at<float>(idX, idY);
		}

	}

}
