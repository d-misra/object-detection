INCL_OPENCV = $(shell pkg-config --cflags --libs opencv)
INCL_BOOST = -lboost_system -lboost_filesystem

task1: 
	g++ $(INCL_OPENCV) $(INCL_BOOST) task1.cpp -o task1