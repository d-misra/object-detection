#include "HOG.cpp"
#include "RandomForrest.cpp"
#include "Dataset.cpp"

// Boost File System
namespace bfs = boost::filesystem;

namespace tdcv {
    namespace helpers {
        void load_dataset(tdcv::HOG& hog, bfs::path data_path, int n_classes, tdcv::Dataset& dataset) {
            std::vector<float> descriptors;

            for (int i = 0; i < n_classes; i++)
            {
                // intializing the class folder
                bfs::path classFolder = data_path / ("0" + std::to_string(i));
                bfs::directory_iterator classIterator{classFolder};

                // for each class calculate the hog and add them together the corresponding label
                while(classIterator != bfs::directory_iterator{}) {
                    // Get image file path
                    bfs::path imageFile = *classIterator++;
                    
                    // Read the image
                    cv::Mat image = cv::imread(imageFile.c_str(), CV_LOAD_IMAGE_COLOR);
                    
                    // Compute the hog features
                    hog.computeHOG(image, descriptors);
                    
                    //forming mat1f array to add the values to feats matrix,basically type casting the descriptors
                    cv::Mat1f descriptors_mat(1, descriptors.size(), descriptors.data());

                    // Add image to the training set
                    dataset.push_back(descriptors_mat, i);
                }
            }
        }
    };
};