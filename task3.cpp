#include <iostream>
#include <fstream>

#include <string>

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "Logger.h"
#include "HOG.cpp"
#include "RandomForrest.cpp"
#include "SVM.cpp"
#include "Dataset.cpp"
#include "RegionProposal.cpp"
#include "helpers.cpp"
#include "external/nms.h"
#include "external/glob.h"

// Boost File System
namespace bfs = boost::filesystem;

static void help(std::string proc_name) {
    std::cout << "Usage:" << std::endl <<
    proc_name << " data_path (confidence_threshold)" << std::endl <<
    "-- data_path: a folder where there is task2/train and task2/test folders" << std::endl <<
    "-- confidence_threshold: classifier operating confidence threshold" << std::endl;
}

/*
* Get Padded Region of Interest
* Source: https://stackoverflow.com/a/42032814
*/
cv::Mat getPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height, cv::Scalar paddingColor) {
    int bottom_right_x = top_left_x + width;
    int bottom_right_y = top_left_y + height;

    cv::Mat output;
    if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {
        // border padding will be required
        int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

        if (top_left_x < 0) {
            width = width + top_left_x;
            border_left = -1 * top_left_x;
            top_left_x = 0;
        }
        if (top_left_y < 0) {
            height = height + top_left_y;
            border_top = -1 * top_left_y;
            top_left_y = 0;
        }
        if (bottom_right_x > input.cols) {
            width = width - (bottom_right_x - input.cols);
            border_right = bottom_right_x - input.cols;
        }
        if (bottom_right_y > input.rows) {
            height = height - (bottom_right_y - input.rows);
            border_bottom = bottom_right_y - input.rows;
        }

        cv::Rect R(top_left_x, top_left_y, width, height);
        copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, cv::BORDER_CONSTANT, paddingColor);
    }
    else {
        // no border padding required
        cv::Rect R(top_left_x, top_left_y, width, height);
        output = input(R);
    }
    return output;
}

void save_cache_selective_search(const std::vector<cv::Rect>& proposals, std::string name) {
    std::ofstream cache_file;
    cache_file.open("cache/selective_search_" + name + ".txt");
    for(int l = 0; l < proposals.size(); l++) {
        cache_file  << proposals[l].x <<
                " " << proposals[l].y <<
                " " << proposals[l].width <<
                " " << proposals[l].height << std::endl;
    }
    cache_file.close();
}

void load_cache_selective_search(std::vector<cv::Rect>& proposals, std::string name) {
    std::ifstream cache_file;
    cache_file.open("cache/selective_search_" + name + ".txt");
    if(cache_file.is_open()) {
        std::string line;
        std::vector<std::string> line_split;
        while(std::getline(cache_file, line)) {
            boost::split(line_split, line, [](char c){return c == ' ';});
            cv::Rect bbox(
                stoi(line_split[0]),
                stoi(line_split[1]),
                stoi(line_split[2]),
                stoi(line_split[3])
            );
            proposals.push_back(bbox);
        }
    }
    cache_file.close();
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::debug);

	if (argc < 2) {
        help(argv[0]);
        return -1;
    }

    const bfs::path data_path(argv[1]);

	if (!bfs::is_directory(data_path)) {
		logger->error("Error - %s is not a valid directory!", data_path.c_str());
		return 1;
	}

    const int n_classes = 4;
    const int n_trees = 200;
    float confidence_threshold = 0.7;
    const bool use_cache = true;

    if (argc == 3) {
        logger->warn("Set confidence threshold to {}", std::stof(argv[2]));
        logger->warn("Set confidence threshold to {}", argv[2]);
        confidence_threshold = std::stof(argv[2]);
    }

    // HOG
    tdcv::HOG hog;

    // Datasets
    tdcv::Dataset training_set(n_classes);

    // Classifier
    tdcv::RandomForrest classifier(n_trees, n_classes);
    // tdcv::SVM classifier;

    // Load Training & Testing Datasets
    // logger->info("Loading datasets ...");
    // tdcv::helpers::load_dataset(hog, data_path / "task3" / "train", n_classes, training_set);

    // Training the classifier
    // logger->info("Training classifier ...");
    // classifier.train(training_set);

    logger->info("Loading saved classifier ...");
    classifier.load("models/exp_deep_rf_128_*");

    // Region Proposal
    tdcv::RegionProposal selective_search;

    // Test files
    auto test_files = external::glob((data_path / "task3" / "test" / "*").c_str());
    
    // Prediction Variables
    int proposal_label;
    float proposal_confidence;

    for(int test_idx = 0; test_idx < test_files.size(); test_idx++) {
        // intializing the test folder
        std::vector<cv::Rect> box_proposals;
        std::vector<std::vector<cv::Rect> > box_proposals_per_label(n_classes);
        std::vector<std::vector<float> > box_proposals_confidence_per_label(n_classes);
        std::vector<cv::Rect> box_proposals_valid;
        std::vector<float> box_proposals_confidence;
        std::vector<float> descriptors;
        
        // Get image file path
        bfs::path imageFile(test_files[test_idx]);
        
        // Read the image
        logger->info("Loading Image ...");
        cv::Mat image = cv::imread(imageFile.c_str(), CV_LOAD_IMAGE_COLOR);

        logger->debug("Image Size: ({}, {})", image.size().height, image.size().width);

        // Visualization Image
        cv::Mat visualization_image = image.clone();

        /*
        * Selective Search Implementation
        */
        // Propose Regions

        if(use_cache) {
            logger->info("Loading cached region proposals ...");
            load_cache_selective_search(box_proposals, imageFile.stem().c_str());
        } else {
            logger->info("Proposing Regions ...");
            selective_search.propose_regions(image, box_proposals, true);
            
            logger->info("Cache selective search output ...");
            save_cache_selective_search(box_proposals, imageFile.stem().c_str());
        }
        
        logger->debug("box_proposals.size() = {}", box_proposals.size());

        int image_area = image.size().width * image.size().height;

        // Generate Proposals & Extract Features
        logger->info("Extracting features per region ...");
        for(int i = 0; i < box_proposals.size(); i++) {
            descriptors.clear();

            // Filter out bounding boxes that are too big 
            if (box_proposals[i].area() > 0.25 * image_area) {
                logger->debug("Skipping box with area larger than quarter of image size...");
                continue;
            }
            
            // Filter out bounding boxes that are too small
            if (box_proposals[i].area() < 0.01 * image_area) {
                logger->debug("Skipping box with area smaller than one percent of image size...");
                continue;
            }

            // Rectify bounding boxes-- make square bounding boxes!
            int max_dim = std::max(box_proposals[i].width, box_proposals[i].height);
            int min_dim = std::min(box_proposals[i].width, box_proposals[i].height);
            int delta_dim = (max_dim - min_dim) / 2;
            int roi_x, roi_y;
            if (min_dim == box_proposals[i].height) {
                roi_x = box_proposals[i].x;
                roi_y = box_proposals[i].y - delta_dim;
            } else {
                roi_x = box_proposals[i].x - delta_dim;
                roi_y = box_proposals[i].y;
            }

            // Skip those bounding box outside of the image!
            if(
                roi_x < 0 ||                                // Out of left border
                roi_y < 0 ||                                // Out of top border
                roi_x + max_dim > image.size().width ||     // Out of right border
                roi_y + max_dim > image.size().height       // Out of left border
            ) {
                logger->debug("Skipping box out of image borders ...");
                continue;
            }

            // Crop Image Proposal
            // cv::Mat image_proposal = image(box_proposals[i]);
            cv::Mat image_proposal = getPaddedROI(
                image,
                roi_x, roi_y,
                max_dim, max_dim,
                cv::Scalar(0,0,0)
            );

            // Update box proposals to reflect changes
            box_proposals[i].x = roi_x;
            box_proposals[i].y = roi_y;
            box_proposals[i].height = max_dim;
            box_proposals[i].width = max_dim;
            
            // Compute the hog features
            hog.computeHOG(image_proposal, descriptors);
            // hog.visualizeHOG(image_proposal, descriptors, hog.getHogDetector());

            classifier.predict_one(cv::Mat1f(descriptors), proposal_label, proposal_confidence);

            // Harsh Threshold On Confidence
            if(proposal_confidence < confidence_threshold) {
                proposal_label = 3;
            }

            logger->debug("Prediction Label= {} ...", proposal_label);

            // Box Proposals Per Label
            box_proposals_per_label[proposal_label].push_back(box_proposals[i]);
            box_proposals_confidence_per_label[proposal_label].push_back(proposal_confidence);

            if (proposal_label == 0 || proposal_label == 1 || proposal_label == 2) {
                box_proposals_valid.push_back(box_proposals[i]);
                box_proposals_confidence.push_back(proposal_confidence);
            }

            // Font Defaults
            float fontScale = 0.5;
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;

            if (proposal_label == 0) {
                cv::rectangle(
                    visualization_image, 
                    box_proposals[i], 
                    cv::Scalar(255, 0, 0)
                );
                cv::putText(
                    visualization_image, 
                    std::to_string(proposal_confidence),
                    cv::Point(box_proposals[i].x + 5, box_proposals[i].y + 5),
                    fontFace,
                    fontScale,
                    cv::Scalar(255, 0, 0)
                );
            }
            
            if (proposal_label == 1) {
                cv::rectangle(
                    visualization_image, 
                    box_proposals[i], 
                    cv::Scalar(0, 255, 0)
                );
                cv::putText(
                    visualization_image, 
                    std::to_string(proposal_confidence),
                    cv::Point(box_proposals[i].x + 5, box_proposals[i].y + 5),
                    fontFace,
                    fontScale,
                    cv::Scalar(0, 255, 0)
                );
            } 
            
            if (proposal_label == 2) {
                cv::rectangle(
                    visualization_image, 
                    box_proposals[i], 
                    cv::Scalar(0, 0, 255)
                );
                cv::putText(
                    visualization_image, 
                    std::to_string(proposal_confidence),
                    cv::Point(box_proposals[i].x + 5, box_proposals[i].y + 5),
                    fontFace,
                    fontScale,
                    cv::Scalar(0, 0, 255)
                );
            }
        }

        // Save Detections Before NMS
        logger->info("Saving detection results ...");
        // cv::imshow("Output", visualization_image);
        cv::imwrite(
            "output/task3/" + 
            std::to_string(confidence_threshold) + 
            "_" + imageFile.stem().c_str() + 
            "_detections_.jpg", 
            visualization_image
        );
        // cv::waitKey();

        
        // Non-Maximum Suppression
        std::vector<std::vector<cv::Rect> > resRectsPerLabel;
        logger->info("Running non-maximum suppression ...");
        // nms2(box_proposals_valid, box_proposals_confidence, resRects, 0.3f);

        for(int l = 0; l < n_classes - 1; l++) {
            // Run NMS against class
            std::vector<cv::Rect> resRects;
            nms2(
                box_proposals_per_label[l],
                box_proposals_confidence_per_label[l],
                resRects, 0.3f
            );
            // nms(
            //     box_proposals_per_label[l],
            //     // box_proposals_confidence_per_label[l],
            //     resRects,
            //     0.3f
            // );
            
            // Save Detected Box Per Class
            resRectsPerLabel.push_back(resRects);
        }

        // Visualize BBoxes After NMS
        visualization_image = image.clone();

        for(int l = 0; l < resRectsPerLabel.size(); l++) {
            for(int r = 0; r < resRectsPerLabel[l].size(); r++) {
                if (l == 0) {
                    cv::rectangle(
                        visualization_image, 
                        resRectsPerLabel[l][r], 
                        cv::Scalar(255, 0, 0)
                    );
                }
                
                if (l == 1) {
                    cv::rectangle(
                        visualization_image, 
                        resRectsPerLabel[l][r], 
                        cv::Scalar(0, 255, 0)
                    );
                } 
                
                if (l == 2) {
                    cv::rectangle(
                        visualization_image, 
                        resRectsPerLabel[l][r], 
                        cv::Scalar(0, 0, 255)
                    );
                }
            }
        }

        // Save Detections After NMS
        logger->info("Saving NMS results ...");
        // cv::imshow("Output", visualization_image);
        cv::imwrite(
            "output/task3/" + 
            std::to_string(confidence_threshold) + 
            "_" +  imageFile.stem().c_str() + 
            "_detections_nms.jpg", 
            visualization_image
        );
        // cv::waitKey();

        // Save Predictions After NMS
        logger->info("Saving NMS predictions ...");
        std::ofstream prediction_file;
        prediction_file.open(
            "output/task3/" + 
            std::to_string(confidence_threshold) + 
            "_" +  imageFile.stem().c_str() + 
            "_predictions.txt"
        );
        for(int l = 0; l < resRectsPerLabel.size(); l++) {
            for(int r = 0; r < resRectsPerLabel[l].size(); r++) {
                prediction_file << l << 
                            " " << resRectsPerLabel[l][r].x <<
                            " " << resRectsPerLabel[l][r].y <<
                            " " << resRectsPerLabel[l][r].width <<
                            " " << resRectsPerLabel[l][r].height << std::endl;
            }
        }
        prediction_file.close();
    }
}