#ifndef RandomForrest_CPP
#define RandomForrest_CPP

#include "RandomForrest.h"
#include "Dataset.cpp"

#include <vector>
#include <string>
#include <set>
#include <opencv2/ml.hpp>

namespace tdcv {
    RandomForrest::RandomForrest(int n_trees, int n_labels, int _maxDepth, int _cvFold, int _minSampleCount) : _nLabels(n_labels) {
        for(int i = 0; i < n_trees; i++) {
            _decision_trees.push_back(cv::ml::DTrees::create());
            
            _decision_trees[i]->setMaxDepth(_maxDepth);
            _decision_trees[i]->setMinSampleCount(_minSampleCount);
            _decision_trees[i]->setCVFolds(_cvFold);
        }
    }

    void RandomForrest::train(Dataset& dataset) {
        cv::Mat1f features;
        cv::Mat labels;

        for(int i = 0; i < _decision_trees.size(); i++) {
            // Remove contents of features/labels from previous iterations
            features.release();
            labels.release();

            printf("RandomForrest::RandomForrest (train) tree= %i\n", i+1);
            dataset.random_subsample(features, labels);
            // dataset.as_matrix(features, labels);

            _decision_trees[i]->train(cv::ml::TrainData::create(
                features,
                cv::ml::ROW_SAMPLE,
                labels
            ));
        }
    }

    void RandomForrest::predict(const cv::Mat1f& features, cv::Mat& predicted_labels, cv::Mat& predicted_confidences) {
        int tree_predictions[_nLabels];
        int prediction;
        float confidence;
        float best_confidence = 0;
        int best_prediction;

        for(int i = 0; i < features.size().height; i++) {
            best_confidence = 0;
            best_prediction = -1;

            // Clear Histogram
            for (int j = 0; j < _nLabels; j++) {
                tree_predictions[j] = 0;
            }

            for (int j = 0; j < _decision_trees.size(); j++) {
                prediction = _decision_trees[j]->predict(features.row(i));
                tree_predictions[prediction]++;
            }

            for (int k = 0; k < _nLabels; k++) {
                confidence = (float)tree_predictions[k] / _decision_trees.size();
                if (best_confidence < confidence) {
                    best_confidence = confidence;
                    best_prediction = k;
                }
            }

            predicted_labels.push_back(best_prediction);
            predicted_confidences.push_back(best_confidence);
        }
    }

    void RandomForrest::predict_one(const cv::Mat1f& features, int &predicted_label, float &predicted_confidence) {
        std::vector<float> tree_predictions(_nLabels, 0);
        int prediction;
        float confidence;

        for (int j = 0; j < _decision_trees.size(); j++) {
            prediction = _decision_trees[j]->predict(features);
            tree_predictions[prediction]++;
        }

        cv::Mat softmax_predictions(tree_predictions);
        softmax_predictions /= _decision_trees.size();

        std::cout << softmax_predictions << std::endl;

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(softmax_predictions, &minVal, &maxVal, &minLoc, &maxLoc);

        predicted_label = maxLoc.y;
        predicted_confidence = maxVal;
    }

    void RandomForrest::save(std::string ForestName) {
        for (int tree_idx = 0; tree_idx < _decision_trees.size(); tree_idx++) {
            _decision_trees[tree_idx]->save(ForestName + std::to_string(tree_idx));
        }
    }

    void RandomForrest::load(std::string ForestName, int nbr_trees) {
        for (int tree_idx = 0; tree_idx < nbr_trees; tree_idx++) {
            _decision_trees[tree_idx] = cv::ml::DTrees::load(ForestName + std::to_string(tree_idx));
        }
    }
};

#endif // RandomForrest_CPP