#ifndef REGION_PROPOSAL_H
#define REGION_PROPOSAL_H

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

namespace tdcv {
    class RegionProposal {
    public:
        RegionProposal(bool use_threads = true, int n_threads = 4);

        void propose_regions(cv::Mat& search_image, std::vector<cv::Rect>& rects, bool fast= false);
        void visualize_regions(cv::Mat& search_image, bool fast= false);

    private:
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> _selectiveSearch;
    };
};

#endif // REGION_PROPOSAL_H
