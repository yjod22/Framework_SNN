# Framework_SNN
A python-based framework is provided in this repository. This framework proposes ways of optimizing a DCNN such that its implementation using SCs has better accuracy. Sparse solutions derived from L1-regularization contribute to increasing the accuracies in MUX-based addition operations. The sparsity can be enhanced by retraining the DCNN with shuffled pre-trained weights.

https://www.researchgate.net/publication/338677860_Master_Thesis_Software_Framework_for_SC-oriented_Deep_Convolutional_Neural_Networks

## Instruction guide for optimized SC-oriented DCNNs
This subsection provides instruction for building, optimizing, and running SC-oriented DCNNs. Python version 3.5 is used in the framework. In the framework, users can design a certain DCNN with the following steps.

### Step 1. Generate global variables
It is recommended to define callback functions of a DCNN in a different python file. User should declare an instance of BNN and the number of the BNN's epochs as the global type in the main file. By using these global variables, the callback functions can refer to the BNN.

### Step 2. Define callback functions
on batch end and on epoch end are essential callback functions that users have to define. Weights scaling should be done in on batch end. Trained parameters should be saved in a specific path every end of epochs. Codes for saving the trained parameters should be specified in on epoch end. 

### Step 3. Define activation functions of a BNN
BNNs refer to Keras library to build a DCNN. Keras library allows users to define their activation functions. Users should define activation functions of a BNN in the main file.

### Step 4. Change the range and format of the input dataset
To make use of stochastic computation, the range of an input dataset has to be set between -1 and +1. Target vectors of the samples have to be converted to binary class matrices. Theano that the framework uses as a back-end engine prefers channels first order in representing images. Therefore, image data format should be set to channels first in Keras JSON file. Lastly, the input shape of a BNN have to be set to (1, X, Y) where (X, Y) is the input format.

### Step 5. Build a BNN
Users can design the specification of a BNN. The instance of BNModel should be declared as a global type. The defined callback functions have to be specified as parameters of BNModel's Fit method.

### Step 6. Optimize the BNN
Users can retrain the BNN by using OptimizeNetwork method of BNModel. OptimizeNetwork requires the following parameters:
testNumber: The name of the main file
titleLastEpochWeight: The path where trained parameters from the last epoch are saved
titleFirstEpochWeight: The path where trained parameters from the first epoch are saved
callbackFunction: The name of the class in which the callback functions of a DCNN are defined
cntIter: The number of the maximum retraining iteration
tupleLayer: The indices of the layers that are supposed to be retrained
x_train, y_train, x_test, y_test: Loaded data from MNIST dataset
epochs: The number of epochs for every retraining
batch_size: The size of the batch

#### The retraining algorithm
![image1](https://github.com/yjod22/Framework_SNN/blob/releasedSW/retrainingAlgorithm.png)

### Step 7. Extract trained parameters
Users can extract the trained parameters. The parameterAdaptive of the method GetConvolutionLayerWeightsBiasesSN has to be set to True to use the adaptive function in MUXs. When Adaptive is set to True, the number of a MUX's input is identical to the number of non-zero
weights in a kernel. 

### Step 8. Iterate over the test samples
To run inference in a SNN, the following steps 9, 10, and 11 have to be iterated over the test samples of the dataset.

### Step 9. Convert BNs to SNs
Users can convert BNs in a test sample data to SNs.

### Step 10. Build & run a SNN
Users can build and run a SNN. 

### Step 11. Check the inference accuracy
Users can check the inference accuracy of the SNN by comparing the output of the SNN and the target vector of the test sample.


##  Test cases
### The simple test case to verify the framework
https://github.com/yjod22/Framework_SNN/blob/releasedSW/test/network_test.py

### The complex test cases to validate the framework
https://github.com/yjod22/Framework_SNN/blob/releasedSW/test/
![image2](https://github.com/yjod22/Framework_SNN/blob/releasedSW/testCaseConfiguration.png)

## The required keras.json file setting
{

    "image_data_format": "channels_first",
    
    "epsilon": 1e-07,    
    
    "floatx": "float32",
    
    "backend": "theano"
    
}

##  The required python packages for the framework
absl-py	0.2.2

astor	0.6.2

blas	1

bleach	1.5.0

certifi	2018.8.24

chardet	3.0.4

decorator	4.2.1

gast	0.2.0

git	2.17.0

grpcio	1.12.0

h5py	2.7.1

hdf5	1.10.1

html5lib	0.9999999

icc_rt	2017.0.4

idna	2.7

intel-openmp	2018.0.0

keras	2.1.6

libgpuarray	0.7.6

libprotobuf	3.5.2

libpython	2.1

m2w64-binutils	2.25.1

m2w64-bzip2	1.0.6

m2w64-crt-git	5.0.0.4636.2595836

m2w64-gcc	5.3.0

m2w64-gcc-ada	5.3.0

m2w64-gcc-fortran	5.3.0

m2w64-gcc-libgfortran	5.3.0

m2w64-gcc-libs	5.3.0

m2w64-gcc-libs-core	5.3.0

m2w64-gcc-objc	5.3.0

m2w64-gmp	6.1.0

m2w64-headers-git	5.0.0.4636.c0ad18a

m2w64-isl	0.16.1

m2w64-libiconv	1.14

m2w64-libmangle-git	5.0.0.4509.2e5a9a2

m2w64-libwinpthread-git	5.0.0.4634.697f757

m2w64-make	4.1.2351.a80a8b8

m2w64-mpc	1.0.3

m2w64-mpfr	3.1.4

m2w64-pkg-config	0.29.1

m2w64-toolchain	5.3.0

m2w64-tools-git	5.0.0.4592.90b8472

m2w64-windows-default-manifest	6.4

m2w64-winpthreads-git	5.0.0.4634.697f757

m2w64-zlib	1.2.8

mako	1.0.7

markdown	2.6.11

markupsafe	1

mkl	2018.0.3

mkl-service	1.1.2

mkl_fft	1.0.1

mkl_random	1.0.1

msys2-conda-epoch	20160418

numpy	1.14.3

numpy-base	1.14.3

pip	9.0.3

plotly	3.2.1

protobuf	3.5.2

pygpu	0.7.6

python	3.5.5

pytz	2018.3

pyyaml	3.12

requests	2.19.1

retrying	1.3.3

scikit-learn	0.20.0

scipy	1.0.1

setuptools	39.0.1

six	1.11.0

tensorboard	1.8.0

tensorflow	1.8.0

tensorflow-base	1.8.0

termcolor	1.1.0

theano	1.0.1

urllib3	1.23

vc	14

vs2015_runtime	14.0.25123

werkzeug	0.14.1

wheel	0.31.0

wincertstore	0.2

yaml	0.1.7

zlib	1.2.11
