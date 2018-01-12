#ifndef DEF_MyForest
#define DEF_MyForest
#include <opencv2/ml.hpp>
///set opencv and c++ namespaces
using namespace cv::ml;
using namespace cv;
using namespace std;
class MyForest{
public:
	MyForest();
	void create(int size_forest, int CVFolds, int MaxDepth, int MinSample_Count, int MaxCategories);
	void train(vector<Mat1f> label_per_feats, Mat labels, int size_samples__per_class[]);
	double * predict(vector<float> test_descriptors);
	void save(string ForestName);
	void load(string ForestName, int nbr_trees);



private:
	Ptr<DTrees> myDTree[100];
	int CVFolds;
	int MaxDepth; 
	int MinSample_Count; 
	int MaxCategories;
	int size_forest;
};

#endif