# TDCV Exercise 2

## Compile The Code

In order to compile the code on a Unix environment, you need to make sure that you install the opencv and a g++ compiler.

Run the following:
```
g++ $(pkg-config --cflags --libs opencv) executable.cpp -o {Executable}
```

Example Compilation of `bbox_generation.cpp`:
```
g++ $(pkg-config --cflags --libs opencv) bbox_generation.cpp -o bbox_generation
```

## Resources:

### BBox Generation: Selective Search
1. https://github.com/watanika/selective-search-cpp
2. https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/
