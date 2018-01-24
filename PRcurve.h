#ifndef PRcurve_H
#define PRcurve_H

#include <opencv2/ml.hpp>
#include <vector>
#include <boost/filesystem.hpp>
#include<iostream>
#include <vector>
#include <climits>
#include <string>

// Boost File System
namespace bfs = boost::filesystem;

namespace tdcv {
    class PRcurve {
    public:
        PRcurve(int n_labels = 1);

        void plot_curve(cv::Mat xdata,cv::Mat ydata);
        void read_gtdata(bfs::path data_path);
        void get_currimg_gtdata(bfs::path data_path);
        void calc_PandR();
        
    private: 
        // NOTE: Index is the label, and the Matrix are the features.
        
        std::vector<cv::Mat> _gtbox;
        std::vector<int*> _class0;
        std::vector<int*> _class1;
        std::vector<int*> _class2;
        std::vector<int*> _curr_img_gt;
        int _true_positives,_false_positives,_false_negatives;
        float _precision,_recall;
    };
};

#endif // Dataset_H