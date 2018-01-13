COMPILER_VERSION = -std=c++11
INCL_OPENCV = $(shell pkg-config --cflags --libs opencv)
INCL_BOOST = -lboost_system -lboost_filesystem

clean:
	rm -f task1
	rm -f task2

task1:
	g++ $(COMPILER_VERSION) $(INCL_OPENCV) $(INCL_BOOST) task1.cpp -o task1

task2:
	g++ $(COMPILER_VERSION) $(INCL_OPENCV) $(INCL_BOOST) task2.cpp -o task2