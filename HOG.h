#ifndef HOG_H
#define HOG_H

#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>

namespace tdcv {
    class HOG {
    public:
        HOG();

        void computeHOG(cv::Mat img, std::vector<float>& descriptors);
        void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor = 3);
        cv::HOGDescriptor& getHogDetector();

    private:  
        cv::Size _winSize;
        cv::Size _blockSize;
        cv::Size _blockStride;
        cv::Size _cellSize;
        int _nbins;
        const int _derivAperture = 1;
        const double _winSigma = -1;
        const int _histogramNormType = cv::HOGDescriptor::L2Hys;
        const double _L2HysThreshold = 0.2;
        const bool _gammaCorrection = false;
        const int _nlevels= cv::HOGDescriptor::DEFAULT_NLEVELS;
        const bool _signedGradient = false;
        
        cv::HOGDescriptor _hog;

        cv::Mat img_gray;
        cv::Mat img_gray_resized;
    };
};

#endif // HOG_H