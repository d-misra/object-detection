#ifndef Logger_H
#define Logger_H

#include <vector>

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

#endif // Logger_H