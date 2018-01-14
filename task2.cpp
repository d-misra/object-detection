#include <iostream>

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include "HOG.cpp"
#include "RandomForrest.cpp"
#include "Dataset.cpp"
#include "external/evaluation.h"
#include "helpers.cpp"

// Boost File System
namespace bfs = boost::filesystem;

static void help() {
    std::cout << "Usage:" << std::endl <<
    "./task2 data_path forrest_name" << std::endl <<
    "-- data_path: a folder where there is task2/train and task2/test folders" << std::endl <<
    "-- forrest_name: a prefix for the forrest to be saved in" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 3) {
        help();
        return -1;
    }

    const bfs::path data_path(argv[1]);

	if (!bfs::is_directory(data_path)) {
		printf("Error - %s is not a valid directory!\n", data_path.c_str());
		return 1;
	}

    const int n_classes = 6;
    const int n_trees = 24;
    
    // HOG
    tdcv::HOG hog;

    // Datasets
    tdcv::Dataset training_set(n_classes);
    tdcv::Dataset testing_set(n_classes);

    // Classifier
    tdcv::RandomForrest classifier(n_trees, n_classes);

    // Testing placeholders
    cv::Mat1f testing_features;
    cv::Mat testing_labels, predicted_labels, predicted_confidences;

    // Load Training & Testing Datasets
    tdcv::helpers::load_dataset(hog, data_path / "task2" / "train", n_classes, training_set);
    tdcv::helpers::load_dataset(hog, data_path / "task2" / "test", n_classes, testing_set);

    printf("Loaded datasets ...\n");

    // Training the classifier
    classifier.train(training_set);
    printf("Trained classifier ...\n");

    // Testing the classifier
    testing_set.as_matrix(testing_features, testing_labels);
    printf("Loaded testing features and labels ...\n");
    
    classifier.predict(testing_features, predicted_labels, predicted_confidences);
    printf("Predicted testing features ...\n");
    
    // Convert Mat labels in to vec<int> labels for evaluation
    std::vector<int> vec_predicted_labels, vec_testing_labels;
    for (int i = 0; i < testing_labels.size().height; i++) {
        vec_predicted_labels.push_back(predicted_labels.at<int>(i, 0));
        vec_testing_labels.push_back(testing_labels.at<int>(i, 0));
    }

    // Evaluate classifier
    external::Confusion confusion(vec_testing_labels, vec_predicted_labels);
    confusion.print();

    external::Evaluation evaluation(confusion);
    evaluation.print();
}