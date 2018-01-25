#ifndef LogReg_CPP
#define LogReg_CPP

#include "LogReg.h"
#include "Logger.h"
#include "Dataset.cpp"

#include <vector>
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

namespace tdcv {
    LogReg::LogReg() {

    }

    void LogReg::train(Dataset& dataset) {
        cv::Mat1f features;
        cv::Mat labels;

        dataset.random_subsample(features, labels);
        // dataset.as_matrix(features, labels);

        labels.convertTo(labels, CV_32F);

        _logreg = cv::ml::LogisticRegression::create();
        _logreg->setLearningRate(0.001);
        _logreg->setIterations(200);
        // _logreg->setRegularization(cv::ml::LogisticRegression::REG_L2);
        _logreg->setTrainMethod(cv::ml::LogisticRegression::BATCH);
        _logreg->setMiniBatchSize(10);
        // features, cv::ml::ROW_SAMPLE, 
        _logreg->train(cv::ml::TrainData::create(
            features,
            cv::ml::ROW_SAMPLE,
            labels
        ));
    }

    void LogReg::predict(const cv::Mat1f& features, cv::Mat& predicted_labels, cv::Mat& predicted_confidences) {
        int label;
        float prob;

        logger->debug("features.size()= ({}, {})", features.row(0).size().height, features.row(0).size().width);
        for (int i = 0; i < features.size().height; i++) {
            predict_one(features.row(i), label, prob);
            predicted_labels.push_back(label);
            predicted_confidences.push_back(prob);
        }
    }

    void LogReg::predict_one(const cv::Mat1f& features, int &predicted_label, float &predicted_confidence) {
        cv::Mat labels;
        predicted_confidence = _logreg->predict(features.reshape(1,1), labels);
        predicted_label = labels.at<int>(0,0);
        std::cout << labels << predicted_label << " " << predicted_confidence << std::endl;
    }

    void LogReg::save(std::string ForestName) {
        // for (int tree_idx = 0; tree_idx < _decision_trees.size(); tree_idx++) {
        //     _decision_trees[tree_idx]->save(ForestName + std::to_string(tree_idx));
        // }
    }

    void LogReg::load(std::string ForestName, int nbr_trees) {
        // for (int tree_idx = 0; tree_idx < nbr_trees; tree_idx++) {
        //     _decision_trees[tree_idx] = cv::ml::DTrees::load(ForestName + std::to_string(tree_idx));
        // }
    }
};

#endif // LogReg_CPP