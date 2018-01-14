#ifndef SVM_H
#define SVM_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>

#include "Dataset.h"

namespace tdcv {
    class SVM {
    public:
        SVM();

        // Save & Load Forrest
        void save(std::string ForestName);
        void load(std::string ForestName, int nbr_trees);

        // Training / Prediction
        void train(Dataset& dataset);
        void predict(const cv::Mat1f& features, cv::Mat& predicted_labels, cv::Mat& predicted_confidences);
        void predict_one(const cv::Mat1f& features, int &predicted_label, float &predicted_confidence);

    private:
        cv::Ptr<cv::ml::SVM> _svm;
    };
};

#endif // SVM_H
