#ifndef Dataset_H
#define Dataset_H

#include <opencv2/ml.hpp>
#include <vector>
#include <boost/filesystem.hpp>

// Boost File System
namespace bfs = boost::filesystem;

namespace tdcv {
    class Dataset {
    public:
        Dataset(int n_labels = 6);

        void push_back(cv::Mat feature, int label);
        void random_subsample(cv::Mat1f& features, cv::Mat& labels);
        void as_matrix(cv::Mat1f& features, cv::Mat& labels);
        int min_features_per_label();

    private: 
        // NOTE: Index is the label, and the Matrix are the features.
        std::vector<cv::Mat> _dataset;
    };
};

#endif // Dataset_H