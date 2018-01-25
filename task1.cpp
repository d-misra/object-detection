#include <iostream>

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include "HOG.cpp"

// Boost File System
namespace bfs = boost::filesystem;

static void help() {
    std::cout << "Usage:" << std::endl <<
    "./task1 data_path" << std::endl <<
    "-- data_path: a folder where there is task2/train and task2/test folders" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
        help();
        return -1;
    }

    const bfs::path data_path(argv[1]);

	if (!bfs::is_directory(data_path)) {
		printf("Error - %s is not a valid directory!\n", data_path.c_str());
		return 1;
	}

    const bfs::path image_file = data_path / "/task1/obj1000.jpg";
    
    cv::Mat image = cv::imread(image_file.c_str(), CV_LOAD_IMAGE_COLOR);
    cv::resize(image, image, cv::Size(WIN_SIZE, WIN_SIZE), 0, 0, cv::INTER_AREA);
    
    std::vector<float> descriptors;

    tdcv::HOG hog;
    hog.computeHOG(image, descriptors);
    hog.visualizeHOG(image, descriptors, hog.getHogDetector());
}