#ifndef Dataset_CPP
#define Dataset_CPP

#include "Dataset.h"

#include <vector>
#include <climits>
#include <string>
#include <opencv2/ml.hpp>
#include <set>
#include <random>


namespace tdcv {
    Dataset::Dataset(int n_labels) {
        for(int i = 0; i < n_labels; i++) {
            _dataset.push_back(cv::Mat());
        }
    }

    void Dataset::push_back(cv::Mat feature, int label) {
        _dataset[label].push_back(feature);
    }

    int Dataset::min_features_per_label() {
        int min_features_per_label = _dataset[0].size().height;

        for (int i = 1; i < _dataset.size(); i++) {
            if(_dataset[i].size().height < min_features_per_label) {
                min_features_per_label = _dataset[i].size().height;
            }
        }

        return min_features_per_label;
    }

    void Dataset::random_subsample(cv::Mat1f& features, cv::Mat& labels) {
        // printf("Dataset::Dataset (random_subsample) :: Start\n");
        
        int randomNum;
        int min_features = min_features_per_label();

        // printf("Dataset::Dataset (random_subsample) min_features= %i\n", min_features);
        
        std::set<int> visited_indices_per_label;

        for (int curr_class = 0; curr_class < _dataset.size(); curr_class++)
        {
            // printf("Dataset::Dataset (random_subsample) _dataset= (%i, %i)\n", _dataset[curr_class].size().height, _dataset[curr_class].size().width);
            visited_indices_per_label.clear();

            // Random Number Generator
            std::mt19937 rng;
            rng.seed(std::random_device()());
            std::uniform_int_distribution<std::mt19937::result_type> dist(
                0,
                _dataset[curr_class].size().height - 1
            ); // distribution in range [0, max_index_per_class]
            
            while(visited_indices_per_label.size() < min_features) {
                randomNum = dist(rng);
                
                if (visited_indices_per_label.count(randomNum) == 0) {
                    visited_indices_per_label.insert(randomNum);

                    // printf("%i, ", randomNum);
                    
                    features.push_back(_dataset[curr_class].row(randomNum));
                    labels.push_back(curr_class);
                }
            }

            // printf("\n");
            // printf("Dataset::Dataset (random_subsample) set[%i].size = %i\n", curr_class, visited_indices_per_label.size());
        }

        // printf("Dataset::Dataset (random_subsample) features= (%i, %i), labels= (%i, %i)\n", 
            // features.size().height, features.size().width,
            // labels.size().height, labels.size().width
        // );

        // printf("Dataset::Dataset (random_subsample) :: End\n");
    }

    void Dataset::as_matrix(cv::Mat1f& features, cv::Mat& labels) {
        for (int curr_class = 0; curr_class < _dataset.size(); curr_class++)
        {
            for(int i = 0; i < _dataset[curr_class].size().height; i++) {
                features.push_back(_dataset[curr_class].row(i));
                labels.push_back(curr_class);
            }
        }
    }
};

#endif // Dataset_CPP