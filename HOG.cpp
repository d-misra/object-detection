#ifndef HOG_CPP
#define HOG_CPP

#include "HOG.h"
#include "Logger.h"

#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
*   Following the qualitative study presented by the author:
*   Source: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
*/
#define CELL_SIZE 12
#define BLOCK_SIZE 2 * CELL_SIZE        // 12
#define BLOCK_STRIDE BLOCK_SIZE / 2     // 6
#define WIN_SIZE BLOCK_SIZE * 2        // 120

namespace tdcv {
    HOG::HOG() : 
        _winSize(cv::Size(WIN_SIZE, WIN_SIZE)), 
        _blockSize(cv::Size(BLOCK_SIZE, BLOCK_SIZE)),
        _blockStride(cv::Size(BLOCK_STRIDE, BLOCK_STRIDE)),
        _cellSize(cv::Size(CELL_SIZE, CELL_SIZE)),
        _nbins (9) {
            logger->debug("HOG :: win_size: {} -- block_size: {} -- block_stride: {} -- cell_size: {}", WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE);
            _hog = cv::HOGDescriptor(_winSize, _blockSize, _blockStride, _cellSize, _nbins,
            _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold,
            _gammaCorrection, _nlevels, _signedGradient);
        }
 
    cv::HOGDescriptor& HOG::getHogDetector() {
        return _hog;
    }

    void HOG::computeHOG(cv::Mat img, std::vector<float>& descriptors) {
        // 1. Convert To Gray Image
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);
        
        // 2. Apply some blur (optional)
        // cv::GaussianBlur(img_gray, img_gray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

        // Scale To Size(120, 120) Image
        cv::resize(img_gray, img_gray_resized, cv::Size(WIN_SIZE, WIN_SIZE), 0, 0, cv::INTER_AREA);

        // Compute HOG
        _hog.compute(img_gray_resized, descriptors);
    }

    void HOG::visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor) {
        cv::Mat visual_image;
        resize(img, visual_image, cv::Size(img.cols * scale_factor, img.rows * scale_factor));


        int n_bins = hog_detector.nbins;
        float rad_per_bin = 3.14 / (float) n_bins;
        cv::Size win_size = hog_detector.winSize;
        cv::Size cell_size = hog_detector.cellSize;
        cv::Size block_size = hog_detector.blockSize;
        cv::Size block_stride = hog_detector.blockStride;


        // prepare data structure: 9 orientation / gradient strenghts for each cell
        int cells_in_x_dir = win_size.width / cell_size.width;
        int cells_in_y_dir = win_size.height / cell_size.height;
        int n_cells = cells_in_x_dir * cells_in_y_dir;
        int cells_per_block = (block_size.width / cell_size.width) * (block_size.height / cell_size.height);

        int blocks_in_x_dir = (win_size.width - block_size.width) / block_stride.width + 1;
        int blocks_in_y_dir = (win_size.height - block_size.height) / block_stride.height + 1;
        int n_blocks = blocks_in_x_dir * blocks_in_y_dir;

        float ***gradientStrengths = new float **[cells_in_y_dir];
        int **cellUpdateCounter = new int *[cells_in_y_dir];
        for (int y = 0; y < cells_in_y_dir; y++) {
            gradientStrengths[y] = new float *[cells_in_x_dir];
            cellUpdateCounter[y] = new int[cells_in_x_dir];
            for (int x = 0; x < cells_in_x_dir; x++) {
                gradientStrengths[y][x] = new float[n_bins];
                cellUpdateCounter[y][x] = 0;

                for (int bin = 0; bin < n_bins; bin++)
                    gradientStrengths[y][x][bin] = 0.0;
            }
        }


        // compute gradient strengths per cell
        int descriptorDataIdx = 0;


        for (int block_x = 0; block_x < blocks_in_x_dir; block_x++) {
            for (int block_y = 0; block_y < blocks_in_y_dir; block_y++) {
                int cell_start_x = block_x * block_stride.width / cell_size.width;
                int cell_start_y = block_y * block_stride.height / cell_size.height;

                for (int cell_id_x = cell_start_x;
                    cell_id_x < cell_start_x + block_size.width / cell_size.width; cell_id_x++)
                    for (int cell_id_y = cell_start_y;
                        cell_id_y < cell_start_y + block_size.height / cell_size.height; cell_id_y++) {

                        for (int bin = 0; bin < n_bins; bin++) {
                            float val = feats.at(descriptorDataIdx++);
                            gradientStrengths[cell_id_y][cell_id_x][bin] += val;
                        }
                        cellUpdateCounter[cell_id_y][cell_id_x]++;
                    }
            }
        }


        // compute average gradient strengths
        for (int celly = 0; celly < cells_in_y_dir; celly++) {
            for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

                float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

                // compute average gradient strenghts for each gradient bin direction
                for (int bin = 0; bin < n_bins; bin++) {
                    gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
                }
            }
        }


        for (int celly = 0; celly < cells_in_y_dir; celly++) {
            for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
                int drawX = cellx * cell_size.width;
                int drawY = celly * cell_size.height;

                int mx = drawX + cell_size.width / 2;
                int my = drawY + cell_size.height / 2;

                rectangle(visual_image,
                        cv::Point(drawX * scale_factor, drawY * scale_factor),
                        cv::Point((drawX + cell_size.width) * scale_factor,
                                    (drawY + cell_size.height) * scale_factor),
                        CV_RGB(100, 100, 100),
                        1);

                for (int bin = 0; bin < n_bins; bin++) {
                    float currentGradStrength = gradientStrengths[celly][cellx][bin];

                    if (currentGradStrength == 0)
                        continue;

                    float currRad = bin * rad_per_bin + rad_per_bin / 2;

                    float dirVecX = cos(currRad);
                    float dirVecY = sin(currRad);
                    float maxVecLen = cell_size.width / 2;
                    float scale = scale_factor / 5.0; // just a visual_imagealization scale,

                    // compute line coordinates
                    float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                    float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                    float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                    float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                    // draw gradient visual_imagealization
                    line(visual_image,
                        cv::Point(x1 * scale_factor, y1 * scale_factor),
                        cv::Point(x2 * scale_factor, y2 * scale_factor),
                        CV_RGB(0, 0, 255),
                        1);

                }

            }
        }


        for (int y = 0; y < cells_in_y_dir; y++) {
            for (int x = 0; x < cells_in_x_dir; x++) {
                delete[] gradientStrengths[y][x];
            }
            delete[] gradientStrengths[y];
            delete[] cellUpdateCounter[y];
        }
        delete[] gradientStrengths;
        delete[] cellUpdateCounter;
        cv::imshow("HOG vis", visual_image);
        cv::waitKey(-1);
        cv::imwrite("hog_vis.jpg", visual_image);
    }
};

#endif // HOG_CPP