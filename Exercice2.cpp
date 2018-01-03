#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string> 
#include "MyForest.h"

///set opencv and c++ namespaces
using namespace cv::ml;
using namespace cv;
using namespace std;

void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor = 3);
/*
* img - the image used for computing HOG descriptors. **Attention here the size of the image should be the same as the window size of your cv::HOGDescriptor instance **
* feats - the hog descriptors you get after calling cv::HOGDescriptor::compute
* hog_detector - the instance of cv::HOGDescriptor you used
* scale_factor - scale the image *scale_factor* times larger for better visualization
*/


void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor) {

	cv::Mat visual_image;
	cv::resize(img, visual_image, cv::Size(img.cols * scale_factor, img.rows * scale_factor));

	int n_bins = hog_detector.nbins;
	float rad_per_bin = 3.14 / (float)n_bins;
	cv::Size win_size = hog_detector.winSize;
	cv::Size cell_size = hog_detector.cellSize;
	cv::Size block_size = hog_detector.blockSize;
	cv::Size block_stride = hog_detector.blockStride;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = win_size.width / cell_size.width;
	int cells_in_y_dir = win_size.height / cell_size.height;
	int n_cells = cells_in_x_dir * cells_in_y_dir;
	int cells_per_block = (block_size.width / cell_size.width) * (block_size.height / cell_size.height);

	int blocks_in_x_dir = (win_size.width - block_size.width) / block_stride.width + 1;
	int blocks_in_y_dir = (win_size.height - block_size.height) / block_stride.height + 1;
	int n_blocks = blocks_in_x_dir * blocks_in_y_dir;

	float ***gradientStrengths = new float **[cells_in_y_dir];
	int **cellUpdateCounter = new int *[cells_in_y_dir];
	for (int y = 0; y < cells_in_y_dir; y++) {
		gradientStrengths[y] = new float *[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x < cells_in_x_dir; x++) {
			gradientStrengths[y][x] = new float[n_bins];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin < n_bins; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}


	// compute gradient strengths per cell
	int descriptorDataIdx = 0;


	for (int block_x = 0; block_x < blocks_in_x_dir; block_x++) {
		for (int block_y = 0; block_y < blocks_in_y_dir; block_y++) {
			int cell_start_x = block_x * block_stride.width / cell_size.width;
			int cell_start_y = block_y * block_stride.height / cell_size.height;

			for (int cell_id_x = cell_start_x;
				cell_id_x < cell_start_x + block_size.width / cell_size.width; cell_id_x++)
				for (int cell_id_y = cell_start_y;
					cell_id_y < cell_start_y + block_size.height / cell_size.height; cell_id_y++) {

				for (int bin = 0; bin < n_bins; bin++) {
					float val = feats.at(descriptorDataIdx++);
					gradientStrengths[cell_id_y][cell_id_x][bin] += val;
				}
				cellUpdateCounter[cell_id_y][cell_id_x]++;
			}
		}
	}


	// compute average gradient strengths
	for (int celly = 0; celly < cells_in_y_dir; celly++) {
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin < n_bins; bin++) {
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}


	for (int celly = 0; celly < cells_in_y_dir; celly++) {
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
			int drawX = cellx * cell_size.width;
			int drawY = celly * cell_size.height;

			int mx = drawX + cell_size.width / 2;
			int my = drawY + cell_size.height / 2;

			rectangle(visual_image,
				cv::Point(drawX * scale_factor, drawY * scale_factor),
				cv::Point((drawX + cell_size.width) * scale_factor,
				(drawY + cell_size.height) * scale_factor),
				CV_RGB(100, 100, 100),
				1);

			for (int bin = 0; bin < n_bins; bin++) {
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				if (currentGradStrength == 0)
					continue;

				float currRad = bin * rad_per_bin + rad_per_bin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = cell_size.width / 2;
				float scale = scale_factor / 5.0; // just a visual_imagealization scale,

												  // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(visual_image,
					cv::Point(x1 * scale_factor, y1 * scale_factor),
					cv::Point(x2 * scale_factor, y2 * scale_factor),
					CV_RGB(0, 0, 255),
					1);

			}

		}
	}


	for (int y = 0; y < cells_in_y_dir; y++) {
		for (int x = 0; x < cells_in_x_dir; x++) {
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;
	cv::imshow("HOG vis", visual_image);
	cv::waitKey(-1);
	cv::imwrite("hog_vis.jpg", visual_image);

}
int main()
{
	///Variables(careful, not all of them)
	Mat src, src_gray, src_gray_resized;
	Mat grad;
	Mat features;
	Mat labels;
	Mat test;
	char* window_name = "HOG Descriptor";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	vector<float> descriptors;

	/// Number of pictures
	///Final values
	int number_of_cat = 6;
	int nbr_pictures_cat[] = { 49, 66, 42, 53, 67, 110 };
	///Temp values
	//int number_of_cat = 2;
	//int nbr_pictures_cat[] = { 2, 2};
	/// Beginning and end of path to pictures
	string schemetrain = "C:/Users/Kamel GUERDA/Desktop/TDCV_exercice2/data/task2/train/";
	string schemetest = "C:/Users/Kamel GUERDA/Desktop/TDCV_exercice2/data/task2/test/";
	string extension = ".jpg";

	///get in right category/fold
	int image_cat;
	int image_index;
	string s_image_cat;
	string s_image_index;
	string path;

	///Create DTrees
	Ptr<DTrees> myDTree[5];// test succeeded, possible to instantiate multiple DTrees
	myDTree[0] = DTrees::create();

	///Set some parameters of the 1st DTree	
	myDTree[0]->setCVFolds(0); // the number of cross-validation folds
	myDTree[0]->setMaxDepth(8);
	myDTree[0]->setMinSampleCount(2);
	//

	///Get HOG descriptor of each picture and create two Mat (one with descriptors, second with labels)
	for (image_cat = 0; image_cat < number_of_cat; image_cat = image_cat+ 1) {
		for (image_index = 0; image_index < nbr_pictures_cat[image_cat]; image_index = image_index + 1) {

			///get 
			s_image_cat = to_string(image_cat);
			while (s_image_cat.length() <2) {
				s_image_cat = "0" + s_image_cat;
			}

			///get a specific image
			s_image_index = to_string(image_index);
			while (s_image_index.length() <4) {
				s_image_index = "0" + s_image_index;
			}
			///create complete path for one specific image
			path = schemetrain  + s_image_cat + "/" + s_image_index + extension;
			///example path="C:/Users/Kamel GUERDA/Desktop/TDCV_exercice2/data/task2/train/00/0000.jpg";
			src = imread(path);

			///check if picture exist
			if (!src.data)
			{
				return -1;
			}

			///Apply some blur 
			//GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
			/// Convert it to gray
			cv::cvtColor(src, src_gray, CV_BGR2GRAY);
			
			///Scale to 120*120 image
			cv::resize(src_gray, src_gray_resized, Size(120, 120), 0, 0, INTER_AREA);

			HOGDescriptor hog(
				src_gray_resized.size(),//Size(20, 20), //winSize
				Size(20, 20), //blocksize
				Size(10, 10), //blockStride,
				Size(10, 10), //cellSize,
				9, //nbins,
				1, //derivAper,
				-1, //winSigma,
				0, //histogramNormType,
				0.2, //L2HysThresh,
				0,//gammal correction,
				64,//nlevels=64
				1);

			hog.compute(src_gray_resized, descriptors, Size(136, 136), Size(8, 8));
			//visualizeHOG(src_gray_resized, descriptors, hog, 6);
			//waitKey(0);
			Mat1f m1(1, descriptors.size(), descriptors.data());
			features.push_back(m1);
			labels.push_back(image_cat);
		}
	}
	
	///Train the DTree with all the data we have
	myDTree[0]->train(ml::TrainData::create(features, ml::ROW_SAMPLE, labels));


	///Test a picture
	///

	path = schemetest + "01" + "/" + "0071" + extension;
	///example path="C:/Users/Kamel GUERDA/Desktop/TDCV_exercice2/data/task2/train/00/0000.jpg";
	src = imread(path);

	///check if picture exist
	if (!src.data)
	{
		return -1;
	}
	/// Convert it to gray
	cv::cvtColor(src, src_gray, CV_BGR2GRAY);
	///Scale to 120*120 image
	cv::resize(src_gray, src_gray_resized, Size(120, 120), 0, 0, INTER_AREA);


	HOGDescriptor hog(
		src_gray_resized.size(),//Size(20, 20), //winSize
		Size(20, 20), //blocksize
		Size(10, 10), //blockStride,
		Size(10, 10), //cellSize,
		9, //nbins,
		1, //derivAper,
		-1, //winSigma,
		0, //histogramNormType,
		0.2, //L2HysThresh,
		0,//gammal correction,
		64,//nlevels=64
		1);
	hog.compute(src_gray_resized, descriptors, Size(136, 136), Size(8, 8));
	Mat1f m2(1, descriptors.size(), descriptors.data());
	test.push_back(m2);

	///Prediction from one tree for one picture
	int prediction;
	prediction = myDTree[0]->predict(test);

	MyForest testForest;
	testForest.create(20);
	testForest.train(features,labels,20);
	int results;
	results = testForest.predict(descriptors);
	//int classificationForest = results[0];

	//int percentageForest = results[1];
	return 0;

}	


///Previous useless stuff used to learn

	/// Generate grad_x and grad_y
	//Mat grad_x, grad_y;
	//Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	///Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	///Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	//Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate), need euclidian distance
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	//grad = abs_grad_x*0.5 + abs_grad_y*0.5;
	/// Create window
	//namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	//imshow(window_name, src_gray_resized);
	//waitKey(0);

