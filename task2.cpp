//this is program to object classfication using hog and random forest
//for trees decision trees are
//the create,train,predict function of the random forest are written in the program itself

//pending things 
//============================
//for task2 to train the random forest, random subset of the data to be generated
//that part is not implemented because of some clarifiction. [15-30 mins]

//things to consider b4 intergarting with task3
//========================================
//change the file path according to your system
//my hog is calculate at the 96x96, hence the patch from the bounding box should be resized to 96x96
//inherently the code will the resize the path to 96 x 96 the image/patch.

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "hog_visualization.cpp"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

vector<float> calc_hog_desc(Mat image, String imgFile, vector<float>descriptorsValues)
{
    /*int scale_factor = 4;
    //String imgFile = imagePath +"train/" +folderName +"/" + imageName;
    //String imgFile = imagePath +folderName +"/" + imageName;
    cv::Mat image = cv::imread(imgFile, CV_LOAD_IMAGE_COLOR);
    std::cout << "\nprocessing image " << imgFile; 
    */
    if (!image.empty())
    {
        //imshow("window", image);
        cv::resize(image, image, Size(96, 96));
        //cv::cvtColor(image, image, CV_BGR2GRAY);
        //cv::namedWindow("image 1", cv::WINDOW_NORMAL);
        //cv::imshow("image 1", image);
        cv::HOGDescriptor hog(Size(96, 96), Size(24, 24), Size(24, 24), Size(12, 12), 9);
        //hog.compute(image, descriptorsValues, Size(0, 0), Size(0, 0), locations);
        hog.compute(image, descriptorsValues, Size(0, 0), Size(0, 0));
        //visualizeHOG(image, descriptorsValues, hog, scale_factor);
        //cv::waitKey(5);
    }
    else
    {
        std::cout << "\nError while reading image " << imgFile <<"\n" ;
        getchar();
    }
    return (descriptorsValues);
}


int main()
{
    Mat1f feats;//to store the hog feature
    Mat labels;//to store the labels or class
    int tot_classes = 6;
	int pic_per_class[] = { 49, 67, 42, 53, 67, 110 }; //number of samples per class/labels
    String imagePath("/home/bala/Documents/studies/tracking/exercise2/data/task2/");
    int scale_factor = 4;
    std::vector<float> descriptorsValues; //used to hog descriptors
    //std::vector<Point> locations;

    //logic is iterate through all the folders and files, then calculate the hog and store it in feats
    for (int i = 0; i < tot_classes; i++) 
    {   
        String folderName="0"+std::to_string(i);//intializing the class folder
        
        //for each class calculate the hog and add them together the corresponding label
        for (int j=0; j < pic_per_class[i] ; j++  )    
        {
            String imageName;
            
            if (j<10) //logic to form the image like 0001.jpg 0r 0010.jpg ot 0100.jpg
            imageName = "000"+std::to_string(j)+".jpg";
            else if (j<100)
            imageName = "00"+std::to_string(j)+".jpg";
            else
            imageName = "0"+std::to_string(j)+".jpg";

            String imgFile = imagePath +"train/" +folderName +"/" + imageName;
            int scale_factor = 4;
            //String imgFile = imagePath +"train/" +folderName +"/" + imageName;
            //String imgFile = imagePath +folderName +"/" + imageName;
            cv::Mat image = cv::imread(imgFile, CV_LOAD_IMAGE_COLOR);
            std::cout << "\nprocessing image " << imgFile; 
            //gng to calculate hog using the below function,the function only return descriptors
            descriptorsValues = calc_hog_desc(image,imgFile,descriptorsValues);
            
            //forming mat1f array to add the values to feats matrix,basically type casting the descriptors
            Mat1f hog_descp(1, descriptorsValues.size(), descriptorsValues.data());
            //Mat1f hog_descp(1,1,descriptorsValues);
            //cout << "size " << descriptorsValues.size() << "\n";
            //cout << "data " << *(descriptorsValues.data()) << "\n";
            
            feats.push_back(hog_descp);      // append at bottom
            labels.push_back(i); // an integer, this is, what you get back in the prediction
        }
    }
    
    //following code to create the forest
    int num_of_dtrees=1;//total number of trees in forest
    Ptr<cv::ml::DTrees>  tree[num_of_dtrees];
    cout<<"\n";
    for (int idx =0 ; idx < num_of_dtrees; idx++ )
    {
    std::cout << "\nCreating tree "<< idx;
    tree[idx] = cv::ml::DTrees::create();
    //setting model parameters
    tree[idx]->setMaxDepth(20);
	tree[idx]->setMinSampleCount(5);
    tree[idx]->setCVFolds(0);
    tree[idx]->setMaxCategories(tot_classes);
    }
    //end of forrest creation

    //gng to train the forest
    cout<<"\n";
    for (int idx=0;idx <num_of_dtrees;idx++)
    {  
        std::cout << "\ntraining tree "<< idx;
        //code pending to create random subsets
        tree[idx]->train(cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels));
    }
    //end of code to train the forest

    // loading test data 

    int test_class = 6;
    int img_per_test_class=10;
    String imageName;    
    for (int i = 0; i < test_class; i++)//looping through all the classes
    {   
        std::cout << "\n\npredicting class "<< i <<"\n";
        String folderName="0"+std::to_string(i);
        float correct,wrong;
        correct=wrong=0;

        for (int j=0;j<img_per_test_class;j++)
        {
            int curr_image=pic_per_class[i]+j;
         
            if (curr_image<100)
            imageName = "00"+std::to_string(pic_per_class[i]+j)+".jpg";
            else if (curr_image<1000)
            imageName = "0"+std::to_string(pic_per_class[i]+j)+".jpg";

            String test_fileName= imagePath + "test/"+ folderName+"/" + imageName;

            //cout <<"\n folderName = "<< folderName;
            //cout <<"\n imageName = " << imageName;
            //cout <<"\n test file = "<<  test_fileName;

            vector<float> test_descrip;
            Mat1f test_feats;
            cv::Mat testimage = cv::imread(test_fileName, CV_LOAD_IMAGE_COLOR);
            std::cout << "\nprocessing image " << test_fileName; 
            test_descrip = calc_hog_desc(testimage,test_fileName,test_descrip);
            Mat1f hog_descp(1, test_descrip.size(), test_descrip.data());
            test_feats.push_back(hog_descp);
            Mat1f predictd_array;
            

            //code to predict a particular hog_feature with all the forest
            int curr_predictd;
            
            int predictd_class[tot_classes];
            std::memset(predictd_class, 0, sizeof (predictd_class));
            //predictd_class is used the number of times each class is predicted by the trees in- 
            //- the forest

            for (int tree_idx=0;tree_idx <num_of_dtrees;tree_idx++)
            {
                curr_predictd=tree[tree_idx]->predict(test_feats,predictd_array);

                //cout<<"\ncurrent predicted "<< curr_predictd;
                //cout<<"\npredicted class before updating "<< predictd_class[curr_predictd];
                predictd_class[curr_predictd]++; 
                //cout<<"\npredicted class after updating "<< predictd_class[curr_predictd];                                           
            }
            //gng to find the class which is predicted maximum times
            int max_predicted_class=tot_classes+1;//assign a class/label outside of the catrgory to debug -
            //incase of error
            for (int class_idx=0;class_idx < (tot_classes-1);class_idx++)
            {
                if (predictd_class[class_idx]<predictd_class[class_idx+1])
                    max_predicted_class=class_idx+1;
                else if (predictd_class[class_idx]>predictd_class[class_idx+1])
                    max_predicted_class=class_idx;

            }
            std::cout << "\nthe maximum predicted class by forest is  "<< max_predicted_class;
            
            float confidence= ((float)predictd_class[max_predicted_class]/(float)num_of_dtrees)*100;
            std::cout << "\nthe confidence of the forest in predicting class is "<< confidence;
            //end of the code to find the class which is predicted maximum times

            //end of code to predict the image with the forest

            //gng to analyze the performance of the forest in predicting particular class 
            
            //calculating the accuracy of the prediction
            if (max_predicted_class == i)//comparing predicted tree result to the class
            {  correct++;}
                //std::cout<<"\n correct "<<correct;}
            else
                wrong++;
        
            //end of code to analyze the performance of the forest in predicting particular class
            

        }  
        
        float accuracy=float(correct/(correct+wrong))*100;
        std::cout << "\nthe accuracy in predicting class is "<< accuracy<<"%";
        //printf("%g", accuracy) ;
        //cout<<"\n correct "<<correct;
        
    }
    int a;
    std::cin >> a;
    return 0;
}