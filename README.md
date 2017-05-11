# Fast Implementation of Online Dictionary Learning for Sparse Coding
Usage
------------
To clone the project and all submodules:
```bash
git clone --recursive https://github.com/d-acharya/OnlineDictionaryLearning.git
```
To update all submodules:
```bash
git submodule foreach git pull origin master
```
To compile:
```bash
mkdir build
cd build 
cmake ..
make
./test_odl
```

Existing Implementations
------------
**C++ Implementation**
[SPAM Webpage](http://spams-devel.gforge.inria.fr/downloads.html)


**Java Implementation**
[Github](https://github.com/maciejkula/dictionarylearning)

**Matlab Implementation**
[Github](https://github.com/tiepvupsu/DICTOL)

**C++ LARS Implementation**
[Github](https://github.com/varung/larscpp/tree/master/src)

ToDos
------------
* Test Script:
  * For testing, see python implementation [dictionary learnning](https://github.com/d-acharya/OnlineDictionaryLearning/blob/master/dict_learning.py) and [image denoising](https://github.com/d-acharya/OnlineDictionaryLearning/blob/master/plot_image_denoising.py). We still need to create a wrapper to call our C implementation of dictionary learning from above python scripts.
* Complete Implementation:
  * Implementation of LARS (Check FISTA implementation in above [matlab implementation](https://github.com/tiepvupsu/DICTOL/blob/master/utils/fista.m)). Possibly replace LARS with FISTA as done in the matlab implementation.


Denoising Demonstration
------------
* After compilation, to test denoising.cpp:
  * ./applicationName Lenna256.png Lenna256Noisy.png
