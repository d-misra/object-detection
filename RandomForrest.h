#ifndef RandomForrest_H
#define RandomForrest_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>

#include "Dataset.h"

namespace tdcv {
    class RandomForrest {
    public:
        RandomForrest(
            int n_trees = 16, 
            int n_labels = 6, 
            int _maxDepth = 200,
            int _cvFold = 0, 
            int _minSampleCount = 2
        );

        // Save & Load Forrest
        void save(std::string ForestName);
        void load(std::string ForestName);

        // Training / Prediction
        void train(Dataset& dataset);
        void predict(const cv::Mat1f& features, cv::Mat& predicted_labels, cv::Mat& predicted_confidences);
        void predict_one(const cv::Mat1f& features, int &predicted_label, float &predicted_confidence);

    private:  
        std::vector<cv::Ptr<cv::ml::DTrees> > _decision_trees;
        int _nLabels;
    };
};

#endif // RandomForrest_H
