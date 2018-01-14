#include <iostream>

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include "HOG.cpp"
#include "RandomForrest.cpp"
#include "Dataset.cpp"
#include "RegionProposal.cpp"
#include "helpers.cpp"
#include "external/nms.h"

// Boost File System
namespace bfs = boost::filesystem;

static void help() {
    std::cout << "Usage:" << std::endl <<
    "./task3 data_path" << std::endl <<
    "-- data_path: a folder where there is task2/train and task2/test folders" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
        help();
        return -1;
    }

    const bfs::path data_path(argv[1]);

	if (!bfs::is_directory(data_path)) {
		printf("Error - %s is not a valid directory!\n", data_path.c_str());
		return 1;
	}

    const int n_classes = 4;
    const int n_trees = 16;

    // HOG
    tdcv::HOG hog;

    // Datasets
    tdcv::Dataset training_set(n_classes);

    // Classifier
    tdcv::RandomForrest classifier(n_trees, n_classes);

    // Load Training & Testing Datasets
    tdcv::helpers::load_dataset(hog, data_path / "task3" / "train", n_classes, training_set);
    printf("Loaded datasets ...\n");

    // Training the classifier
    classifier.train(training_set);
    printf("Trained classifier ...\n");

    // Region Proposal
    tdcv::RegionProposal selective_search;

    // intializing the test folder
    bfs::directory_iterator classIterator{data_path / "task3" / "test"};
    std::vector<cv::Rect> box_proposals;
    std::vector<std::vector<cv::Rect> > box_proposals_per_label(n_classes);
    std::vector<float> descriptors;
    
    // Prediction Variables
    int proposal_label;
    float proposal_confidence;

    while(classIterator != bfs::directory_iterator{}) {
        box_proposals.clear();
        
        // Get image file path
        bfs::path imageFile = *classIterator++;
        
        // Read the image
        printf("Loading Image ...");
        cv::Mat image = cv::imread(imageFile.c_str(), CV_LOAD_IMAGE_COLOR);

        // resize image
        int newHeight = 480;
        int newWidth = image.cols*newHeight/image.rows;
        cv::resize(image, image, cv::Size(newWidth, newHeight));

        // Propose Regions
        printf("Proposing Regions ...");
        selective_search.propose_regions(image, box_proposals, true);

        cv::Mat visualization_image = image.clone();

        // Generate Proposals & Extract Features
        printf("Extracting features per region ...");
        for(int i = 0; i < box_proposals.size(); i++) {
            descriptors.clear();

            // Crop Image Proposal
            cv::Mat image_proposal = image(box_proposals[i]);
            
            // Compute the hog features
            hog.computeHOG(image_proposal, descriptors);
            // hog.visualizeHOG(image_proposal, descriptors, hog.getHogDetector());

            printf("Predicting label per region ...\n");
            classifier.predict_one(cv::Mat1f(descriptors), proposal_label, proposal_confidence);

            // Harsh Threshold On Confidence
            if(proposal_confidence < 0.75) {
                proposal_label = 3;
            }

            printf("Prediction Label= %i ...\n", proposal_label);

            // Box Proposals Per Label
            box_proposals_per_label[proposal_label].push_back(box_proposals[i]);

            if (proposal_label == 0) {
                cv::rectangle(
                    visualization_image, 
                    box_proposals[i], 
                    cv::Scalar(255, 0, 0)
                );
            }
            
            if (proposal_label == 1) {
                cv::rectangle(
                    visualization_image, 
                    box_proposals[i], 
                    cv::Scalar(0, 255, 0)
                );
            } 
            
            if (proposal_label == 2) {
                cv::rectangle(
                    visualization_image, 
                    box_proposals[i], 
                    cv::Scalar(0, 0, 255)
                );
            }
        }

        // printf("BBox :: i= %i, label= %i, conf= %f\n", i, proposal_label, proposal_confidence);
        cv::imshow("Output", visualization_image);
        cv::waitKey();

        // Non-Maximum Suppression
        for(int l = 0; l < n_classes; l++) {
            std::vector<cv::Rect> resRects;
            nms(box_proposals_per_label[l], resRects, 0.3f);
            
            // Reset Preview Image
            visualization_image = image.clone();
            
            for(int i = 0; i < resRects.size(); i++) {
                cv::rectangle(
                    visualization_image,
                    resRects[i],
                    cv::Scalar(255, 0, 0)
                );
            }

            cv::imshow("Output", visualization_image);
            cv::waitKey();
        }
    }
}