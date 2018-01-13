#include <iostream>

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include "HOG.cpp"
#include "RandomForrest.cpp"
#include "Dataset.cpp"
#include "external/evaluation.h"

// Boost File System
namespace bfs = boost::filesystem;

static void help() {
    std::cout << "Usage:" << std::endl <<
    "./task2 data_path forrest_name" << std::endl <<
    "-- data_path: a folder where there is task2/train and task2/test folders" << std::endl <<
    "-- forrest_name: a prefix for the forrest to be saved in" << std::endl;
}

void load_dataset(tdcv::HOG& hog, bfs::path data_path, std::string phase, int n_classes, tdcv::Dataset& dataset) {
    std::vector<float> descriptors;

    for (int i = 0; i < n_classes; i++)
    {
        // intializing the class folder
        bfs::path classFolder = data_path / phase / ("0" + std::to_string(i));
        bfs::directory_iterator classIterator{classFolder};

        // for each class calculate the hog and add them together the corresponding label
        while(classIterator != bfs::directory_iterator{}) {
            // Get image file path
            bfs::path imageFile = *classIterator++;
            
            // Read the image
            cv::Mat image = cv::imread(imageFile.c_str(), CV_LOAD_IMAGE_COLOR);
            
            // Compute the hog features
            hog.computeHOG(image, descriptors);
            
            //forming mat1f array to add the values to feats matrix,basically type casting the descriptors
            cv::Mat1f descriptors_mat(1, descriptors.size(), descriptors.data());

            // Add image to the training set
            dataset.push_back(descriptors_mat, i);
        }
    }
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
    load_dataset(hog, data_path / "task2", "train", 6, training_set);
    load_dataset(hog, data_path / "task2", "test", 6, testing_set);

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

    // printf("testing_labels= %i, predicted_labels= %i ...\n", 
    //     testing_labels.size().height,
    //     predicted_labels.size().height
    // );
    // printf("vec_testing_labels= %i, vec_predicted_labels= %i ...\n", 
    //     vec_testing_labels.size(), 
    //     vec_predicted_labels.size()
    // );

    // cout << "testing_labels= "<< endl << " "  << testing_labels << endl << endl;
    // cout << "vec_testing_labels= "<< endl;
    // for (int i = 0; i < vec_testing_labels.size(); i++) {
    //     cout << vec_testing_labels.at(i) << endl;
    // }

    // Evaluate classifier
    external::Confusion confusion(vec_testing_labels, vec_predicted_labels);
    confusion.print();

    external::Evaluation evaluation(confusion);
    evaluation.print();
}