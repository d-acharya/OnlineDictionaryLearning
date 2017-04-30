# Fast Implementation of Online Dictionary Learning for Sparse Coding


### Existing Implementations
**C++ Implementation**
[SPAM Webpage](http://spams-devel.gforge.inria.fr/downloads.html)


**Java Implementation**
[Github](https://github.com/maciejkula/dictionarylearning)

**Matlab Implementation**
[Github](https://github.com/tiepvupsu/DICTOL)

#### **ToDos**
* Test Script:
  * For testing, see python implementation [dictionary learnning](https://github.com/d-acharya/OnlineDictionaryLearning/blob/master/dict_learning.py) and [image denoising](https://github.com/d-acharya/OnlineDictionaryLearning/blob/master/plot_image_denoising.py). We still need to create a wrapper to call our C implementation of dictionary learning from above python scripts.
* Complete Implementation:
  * Implementation of LARS (Check FISTA implementation in above [matlab implementation](https://github.com/tiepvupsu/DICTOL/blob/master/utils/fista.m)). Possibly replace LARS with FISTA as done in the matlab implementation.
* Pathwise coordinate descent:
  * has surpassed LARS since 2008?
* pseudo code line 4, page 487, http://statweb.stanford.edu/~tibs/ftp/lars.pdf
  * how to find A1?
* matrix inversion function http://www.sanfoundry.com/cpp-program-implement-gauss-jordan-elimination/
