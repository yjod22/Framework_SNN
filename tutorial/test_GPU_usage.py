#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     test_GPU_usage.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   test_GPU_usage.py
# Version:     8.0
# Author/Date: Junseok Oh / 2019-05-23
# Change:      This code is coming from the following website
#			   http://deeplearning.net/software/theano_versions/dev/tutorial/using_gpu.html
# Cause:       Make use of GPU
# Initiator:   Junseok Oh
###############################################################################


# Commands for setup in theano environment
# source /home/username/anaconda3/bin/activate /home/username/anaconda3/envs/snn
# conda install libgpuarray
# conda uninstall mkl=2018
# conda install mkl=2017

# Commands to run the code
# ssh -X ralabXX
# source /home/username/anaconda3/bin/activate /home/username/anaconda3/envs/snn
# THEANO_FLAGS=device=cuda0 /home/username/anaconda3/envs/snn/bin/python /home/username/PycharmProjects/snn/test_GPU_usage.py

from theano import function, config, shared, tensor
import numpy
import time


vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')