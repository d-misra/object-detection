#include <iostream>

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include "Logger.h"
#include "HOG.cpp"
#include "RandomForrest.cpp"
#include "SVM.cpp"
#include "LogReg.cpp"
#include "Dataset.cpp"
#include "external/evaluation.h"
#include "helpers.cpp"

// Boost File System
namespace bfs = boost::filesystem;

static void help(std::string proc) {
    std::cout << "Usage:" << std::endl <<
    proc << " data_path forrest_name n_trees" << std::endl <<
    "-- data_path: a folder where there is task2/train and task2/test folders" << std::endl <<
    "-- forrest_name: a prefix for the forrest to be saved in" << std::endl <<
    "-- n_trees: number of trees" << std::endl;
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::debug);

	if (argc < 4) {
        help(argv[0]);
        return -1;
    }

    const bfs::path data_path(argv[1]);
    const std::string forest_name(argv[2]);

	if (!bfs::is_directory(data_path)) {
		logger->error("Error - %s is not a valid directory!", data_path.c_str());
		return 1;
	}

    const int n_classes = 4;
    const int n_trees = std::stoi(argv[3]);
    
    // HOG
    tdcv::HOG hog;

    // Datasets
    tdcv::Dataset training_set(n_classes);
    tdcv::Dataset testing_set(n_classes);

    // Classifier
    tdcv::RandomForrest classifier(n_trees, n_classes);
    // tdcv::SVM classifier;
    // tdcv::LogReg classifier;

    // Testing placeholders
    cv::Mat1f testing_features;
    cv::Mat testing_labels, predicted_labels, predicted_confidences;

    // Load Training & Testing Datasets
    logger->info("Loading training dataset ...");
    tdcv::helpers::load_dataset(hog, data_path / "task3" / "train", n_classes, training_set);
    
    logger->info("Loading validation dataset ...");
    tdcv::helpers::load_dataset(hog, data_path / "task3" / "val", n_classes, testing_set);

    // Training the classifier
    logger->info("Training classifier ...");
    classifier.train(training_set);

    // Testing the classifier
    logger->info("Loading testing features and labels ...");
    testing_set.as_matrix(testing_features, testing_labels);
    
    logger->info("Predicting labels for testing features ...");
    classifier.predict(testing_features, predicted_labels, predicted_confidences);
    
    // Convert Mat labels in to vec<int> labels for evaluation
    std::vector<int> vec_predicted_labels;
    std::vector<int> vec_testing_labels;
    for (int i = 0; i < testing_labels.size().height; i++) {
        vec_predicted_labels.push_back(predicted_labels.at<int>(i));
        vec_testing_labels.push_back(testing_labels.at<int>(i));
    }

    // Evaluate classifier
    external::Confusion confusion(vec_testing_labels, vec_predicted_labels);
    confusion.print();

    external::Evaluation evaluation(confusion);
    evaluation.print();

    // Save the classifier
    classifier.save(forest_name);
}