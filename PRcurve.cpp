#ifndef PRcurve_CPP
#define PRcurve_CPP

#include "PRcurve.h"

#include<iostream>
#include <vector>
#include <climits>
#include <string>
#include <opencv2/ml.hpp>
#include <boost/filesystem/fstream.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/plot.hpp>
#include <math.h>
#include <cmath>
#include <stdio.h>

namespace tdcv
{
    PRcurve::PRcurve(int n_labels)
    {
        for (int i = 0; i < n_labels; i++)
        {
            _gtbox.push_back(cv::Mat());
        }
        //initializng the constructir for the retriving the ground truth for all the test images
        //has some issues because the boost loader loads the files randomly hence
        // so its hard to track which gt box belongs to which image
        int no_images=44;
        for (int i = 0; i < no_images; i++)
        {
            int temp[4]={0,0,0,0};
            _class0.push_back(temp);
            _class1.push_back(temp);
            _class2.push_back(temp);
            
        }
        //assign values for calculation of PR curves
        _true_positives=0;
        _false_positives=0;
        _false_negatives=0;
        _precision=0;
        _recall=0;
    }

    void PRcurve::get_currimg_gtdata(bfs::path img_path)
    {
        //this function is used to extract the get truth for the current image provided the img path
        bfs::ifstream myfile {img_path};
        int check,label;
        check=0;
        int gt_set[4];
        //std::cout<<"\nfile name = "<<img_path;
        if (myfile.is_open())
        {
            do
            {
                
                //Extract the data from the lines
                myfile >> label;
                myfile >> gt_set[0];
                myfile >> gt_set[1];
                myfile >> gt_set[2];
                myfile >> gt_set[3];
                //move to the next line
                myfile.ignore(20, '\n');
                
                std::cout<<"\nNumbers from the file = "<<gt_set[0]<<" "<<gt_set[1]<<" "<<gt_set[2]
                <<" "<<gt_set[3];
                //cv::Mat curr_gt=cv::Mat(4, 1, CV_16UC, num);
                //_gtbox[label].push_back(curr_gt);
                _curr_img_gt.push_back(gt_set);

                check++;

            }while (check!=3);
            std::cout<<"\n";
            myfile.close();

        }
    }

    void PRcurve::plot_curve(cv::Mat xData,cv::Mat yData)
    {
        //following function is used to plot the graph
        cv::Mat display;
        cv::Ptr<cv::plot::Plot2d> plot;
        //Ptr< Plot2d > 	createPlot2d (Mat dataX, Mat dataY);
        //Ptr< Plot2d > createPlot2d (xData,yData);

        /*
        //some random graph to visulise the data
        float pow1=0.9;
        for (int i = 0; i < 10; ++i)
        {
            xData.at<double>(i) = i*0.1;
            pow1=pow(pow1,2);
            yData.at<double>(i) = pow1;
        }*/
        
        plot= cv::plot::Plot2d::create(xData, yData);
        plot->setPlotSize(1, 1);
        plot->setMaxX(1);
        plot->setMinX(0);
        plot->setMaxY(1);
        plot->setMinY(0);
        plot->setInvertOrientation("true");
        plot->render(display);
        cv::imshow("Plot", display);
        cv::waitKey();

    }

    void PRcurve::calc_PandR(){

        //calculation of precision and recall
        //have some doubts with recall. tomorrow, i will go there and get it confirmed by the guys
        _precision = (float) _true_positives / (float)( _true_positives + _false_positives );
        _recall = (float) _true_positives / (float ) (_true_positives + _false_negatives );
    
    }

    void PRcurve::read_gtdata(bfs::path data_path)
    {
        //this function is used to load all the  ground truth values at once
        //but had some issues as eariler commented
        //would work if we used normal file stream instead of boost
        bfs::path classFolder = data_path / "task3/gt";
        bfs::directory_iterator classIterator{classFolder};
        int img_index=0;
        // for each class calculate the hog and add them together the corresponding label
        while (classIterator != bfs::directory_iterator{})
        {
            // Get image file path
            bfs::path ground_truth_file = *classIterator++;
            bfs::ifstream myfile {ground_truth_file};
            int check,label;
            check=0;
            int num[4];
            std::cout<<"\nfile name = "<<ground_truth_file;
              if (myfile.is_open())
            {
                do
                {
                    //cout << line << '\n';
                    //reading all the values in a particular line
                    myfile >> label;
                    myfile >> num[0];
                    myfile >> num[1];
                    myfile >> num[2];
                    myfile >> num[3];
                    myfile.ignore(20, '\n');
                    
                    std::cout<<"Numbers from the file = "<<num[0]<<" "<<num[1]<<" "<<num[2]<<" "<<num[3]
                    <<"\n";
                    //cv::Mat curr_gt=cv::Mat(4, 1, CV_16UC, num);
                    //_gtbox[label].push_back(curr_gt);
                    if(label == 0)
                    _class0.push_back(num);
                    else if (label == 1)
                    _class1.push_back(num);
                    else if (label == 2)
                    _class2.push_back(num);

                    check++;

                }while ( check!=3);
                
                myfile.close();

            }
            img_index++;
        }
    }

    


};

#endif // PRcurve_CPP