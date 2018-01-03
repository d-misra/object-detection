#include "stdafx.h"
#include "MyForest.h"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml.hpp>
#include <stdlib.h> // srand, rand 
#include <stdio.h>
#include <math.h>
#include <string> 
#include <time.h>       // time 

using namespace std;
using namespace cv;
using namespace cv::ml;



MyForest::MyForest() {
	Ptr<DTrees> myDTree[100];
	int size_forest;
	size_forest = 10;
	for (int i = 0; i < size_forest; i++) {
		myDTree[i] = DTrees::create();
		myDTree[i]->setCVFolds(0); // the number of cross-validation folds
		myDTree[i]->setMaxDepth(8);
		myDTree[i]->setMinSampleCount(2);
	}
}



void MyForest::create(int size_forest) {
	for (int i = 0; i < size_forest; i++) {
		myDTree[i] = DTrees::create();
		myDTree[i]->setCVFolds(0); // the number of cross-validation folds
		myDTree[i]->setMaxDepth(8);
		myDTree[i]->setMinSampleCount(2);
	}
}

void MyForest::train(Mat features, Mat labels, int size_samples) {
	int nbr_rows = features.rows;
	int rand_index;
	int cat;
	Mat temp;
	Mat randomized_features;
	Mat randomized_labels;
	srand(time(NULL));
	for (int i = 0; i < size_forest; i++) {
		for (int j = 0; j < size_samples; j++) {
			rand_index = rand() % nbr_rows;
			temp = features(Range(rand_index, rand_index + 1), cv::Range::all());
			randomized_features.push_back(temp);
			cat = labels.at<int>(rand_index);
			randomized_labels.push_back(cat);
		}
		myDTree[i]->train(ml::TrainData::create(randomized_features, ml::ROW_SAMPLE, randomized_labels));
	}
}


int MyForest::predict(vector<float> descriptors) const {
	Mat1f m2(1, descriptors.size(), descriptors.data());
	Mat test;
	test.push_back(m2);
	int predictions[6] = {0,0,0,0,0,0};
	float temp;
	for (int i = 0; i < size_forest; i++) {
		temp = myDTree[i]->predict(test);
		predictions[(int)temp] = predictions[(int)temp]+1;
	}
	float highest_percentage = 0;
	int cat_predicted;
	for (int i = 0; i<6; i++)
	{
		if (predictions[i] > highest_percentage) {
			highest_percentage = predictions[i];
			cat_predicted = i;
		}
	}
	int complete_answer[2] = { cat_predicted,highest_percentage };
	return cat_predicted;
}
