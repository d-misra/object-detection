#include "HOG.cpp"
#include "RandomForrest.cpp"
#include "Dataset.cpp"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>


// Boost File System
namespace bfs = boost::filesystem;

namespace tdcv {
    namespace helpers {
        void load_dataset(tdcv::HOG& hog, bfs::path data_path, int n_classes, tdcv::Dataset& dataset) {
            std::vector<float> descriptors;
            // add by bala 
            int scaling_factor = 4;
            cv::Mat dst_img, tmp_img;
            cv::namedWindow( "resizing images", CV_WINDOW_AUTOSIZE );


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
                    dataset.push_back(descriptors_mat, i);

                    //code added by bala
                    tmp_img = image;
                    dst_img = tmp_img;

                    //creating image pyramid 
                    for(int i=0; i < scaling_factor; i++)
                    {
                        //using the pyrDown function to scale the image to half of the original size
                        cv::pyrDown( tmp_img, dst_img, cv::Size( tmp_img.cols/2, tmp_img.rows/2 ) );
                        
                        // Compute the hog features
                        hog.computeHOG(dst_img, descriptors);
                        
                        //forming mat1f array to add the values to feats matrix,basically type casting the descriptors
                        cv::Mat1f descriptors_mat(1, descriptors.size(), descriptors.data());

                        // Add image to the training set
                        dataset.push_back(descriptors_mat, i);
                        tmp_img=dst_img;
                        
                        //uncomment to below to visualize the changes in scale of the images
                        //imshow( "resizing images", dst_img );
                        //cv::waitKey();
                    }
                }
            }
        }
    };
};