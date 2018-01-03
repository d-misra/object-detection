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
	void create(int size_forest);
	void train(Mat features, Mat labels, int size_samples);
	int predict(vector<float> descriptors) const;


private:
	Ptr<DTrees> myDTree[100];
	int size_forest;
};

#endif