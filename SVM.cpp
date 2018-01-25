#ifndef SVM_CPP
#define SVM_CPP

#include "SVM.h"
#include "Logger.h"
#include "Dataset.cpp"

#include <vector>
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

namespace tdcv {
    SVM::SVM() {

    }

    void SVM::train(Dataset& dataset) {
        cv::Mat1f features;
        cv::Mat labels;

        // dataset.random_subsample(features, labels);
        dataset.as_matrix(features, labels);

        _svm = cv::ml::SVM::create();
        _svm->setType(cv::ml::SVM::C_SVC);
        _svm->setC(0.1);
        _svm->setKernel(cv::ml::SVM::LINEAR);
        _svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 500, 1e-6));

        _svm->train(features, cv::ml::ROW_SAMPLE, labels);
    }

    void SVM::predict(const cv::Mat1f& features, cv::Mat& predicted_labels, cv::Mat& predicted_confidences) {
        logger->debug("features.size()= ({}, {})", features.row(0).size().height, features.row(0).size().width);
        for (int i = 0; i < features.size().height; i++) {
            predicted_labels.push_back((int)_svm->predict(features.row(i).reshape(1,1)));
            predicted_confidences.push_back(1.f);
        }

        // predicted_labels.push_back(best_prediction);
        // predicted_confidences.push_back(best_confidence);
    }

    void SVM::predict_one(const cv::Mat1f& features, int &predicted_label, float &predicted_confidence) {
        predicted_label = (int)_svm->predict(features.reshape(1,1));
        predicted_confidence = 1.0;
    }

    void SVM::save(std::string ForestName) {
        // for (int tree_idx = 0; tree_idx < _decision_trees.size(); tree_idx++) {
        //     _decision_trees[tree_idx]->save(ForestName + std::to_string(tree_idx));
        // }
    }

    void SVM::load(std::string ForestName, int nbr_trees) {
        // for (int tree_idx = 0; tree_idx < nbr_trees; tree_idx++) {
        //     _decision_trees[tree_idx] = cv::ml::DTrees::load(ForestName + std::to_string(tree_idx));
        // }
    }
};

#endif // SVM_CPP