#ifndef REGION_PROPOSAL_CPP
#define REGION_PROPOSAL_CPP

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

#include "RegionProposal.h"
#include "Logger.h"

namespace tdcv {
    RegionProposal::RegionProposal(bool use_threads, int n_threads) {
        cv::setUseOptimized(use_threads);
        if(use_threads) {
            cv::setNumThreads(n_threads);
        }

        _selectiveSearch = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
    }

    void RegionProposal::propose_regions(cv::Mat& search_image, std::vector<cv::Rect>& bounding_boxes, bool fast) {
        logger->debug("RegionProposal::propose_regions -- Start");

        logger->debug("RegionProposal::propose_regions -- Setting Base Image");
        _selectiveSearch->setBaseImage(search_image);

        logger->debug("RegionProposal::propose_regions -- Setting Fast/Quality Setting");
        if(fast) {
            _selectiveSearch->switchToSelectiveSearchFast();
        } else {
            _selectiveSearch->switchToSelectiveSearchQuality();
        }

        logger->debug("RegionProposal::propose_regions -- Process");
        _selectiveSearch->process(bounding_boxes);

        logger->info("RegionProposal::propose_regions -- End :: bounding_boxes.size()= {}", bounding_boxes.size());
    }

    void RegionProposal::visualize_regions(cv::Mat& search_image, bool fast) {
        std::vector<cv::Rect> bounding_boxes;
        propose_regions(search_image, bounding_boxes, fast);

        cv::Mat visualization_image = search_image.clone();

        for(int i = 0; i < bounding_boxes.size(); i++) {
            cv::rectangle(
                visualization_image, 
                bounding_boxes[i], 
                cv::Scalar(0, 255, 0)
            );
        }

        cv::imshow("Output", visualization_image);
        cv::waitKey();
    }
};

#endif // REGION_PROPOSAL_CPP