#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     test_AdaptiveFunction.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   test_AdaptiveFunction.py
# Version:     9.0
# Author/Date: Junseok Oh / 2019-06-07
# Change:      (SCR_V8.0-1): test adaptive function in STanhSN and STanhLUTSN
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   stanh_testing.py
# Version:     8.0
# Author/Date: Junseok Oh / 2019-05-23
# Change:      (SCR_V6.4-3): Update with tanh(2,3,4,5,6x) 
#			   (SCR_V6.4-9): Update Stanh with LUT for adaptive function
#			   (SCR_V6.4-15): SC-Based Relu verification (failed)
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   stanh_testing.py
# Version:     6.2 (SCR_V6.1-3)
# Author/Date: Junseok Oh / 2019-02-19
# Change:      Define APC_btanhx
# Cause:       Rename the activation functions
# Initiator:   Florian Neugebauer
################################################################################
# File:		   stanh_testing.py
# Version:     6.2 (SCR_V6.1-2)
# Author/Date: Junseok Oh / 2019-02-19
# Change:      Change the input of tanh function
# Cause:       Need to consider the error of BTanh
# Initiator:   Florian Neugebauer
################################################################################
# Version:     6.0
# Author/Date: Junseok Oh / 2018-12-13
# Change:      Implement the APC-based tanh
# Cause:       Replace the stanh with the APC-based tanh
# Initiator:   Florian Neugebauer
################################################################################
# Version:     Initial version
# Author/Date: Florian Neugebauer / 2018-12-06
# Change:      Initial version
# Cause:       Test different stanh circuits
# Initiator:   Dr. Ilia Polian
################################################################################
### test different stanh circuits
import numpy as np
import plotly as py
import plotly.graph_objs as go

def Similarity(x, y):
    sum = 0
    for i in range(len(x)):
        sum += np.square(x[i]-y[i])

    return sum


# Initialize the graphs' data and parameters
numSamples = 1000
reference = np.zeros(numSamples)
reference1 = np.zeros(numSamples)
reference2 = np.zeros(numSamples)
reference3 = np.zeros(numSamples)
reference4 = np.zeros(numSamples)
reference5 = np.zeros(numSamples)

'''
# get some random input values and sort them. These are going to be assigned to the x-axis
values = np.zeros(numSamples)
# values = np.random.random(numSamples)*2 - 1
values = np.random.random(numSamples)
values = np.sort(values, 0)
'''

# get some random input values and sort them. These are going to be assigned to the x-axis
values = np.zeros(numSamples)
values = np.random.random(numSamples)
values = np.sort(values, 0)
for i in range(numSamples):
    values[i] = i/numSamples

# Calculate the graphs' data that are going to be assigned to the y-axis
for i in range(values.shape[0]):
    reference1[i] = min(max(values[i]*1, 0), 1)
    reference2[i] = max(np.tanh(values[i]*0.7), 0)
    reference3[i] = max(np.tanh(values[i]*1), 0)
    reference4[i] = max(np.tanh(values[i]*1.232), 0)
    reference5[i] = max(np.tanh(values[i]*1.84), 0)

# Assign the graphs' data to x-axis and y-axis
relu = go.Scatter(x=values, y=reference1)
tanh0_7x = go.Scatter(x=values, y=reference2)
tanh1x = go.Scatter(x=values, y=reference3)
tanh1_232x = go.Scatter(x=values, y=reference4)
tanh1_84x = go.Scatter(x=values, y=reference5)

# Integrate the graphs' data
data1 = [relu, tanh0_7x, tanh1x, tanh1_232x, tanh1_84x]

# create plots
py.offline.plot(data1, filename='STanh(1x)_ReLU.html')

# Calculate the Similarity of Tanh and ReLU
print('Similarity of Tanh and ReLU')
print(Similarity(reference1, reference1))
print('Similarity of Tanh and ReLU')
print(Similarity(reference1, reference2))
print('Similarity of Tanh and ReLU')
print(Similarity(reference1, reference3))
print('Similarity of Tanh and ReLU')
print(Similarity(reference1, reference4))
print('Similarity of Tanh and ReLU')
print(Similarity(reference1, reference5))


# Calculate the graphs' data that are going to be assigned to the y-axis
scale = [1.21]
sim = [3.37]
i = 0

# Calculate the graphs' data that are going to be assigned to the y-axis
for j in range(values.shape[0]):
    reference1[j] = min(max(values[j]*1, 0), 1)

while (i < 50) :
    for j in range(values.shape[0]):
        reference[j] = max(np.tanh(values[j]*scale[i]), 0)
    sim.append(Similarity(reference1, reference))
    scale.append(scale[i] + 0.001)
    i += 1

# Assign the graphs' data to x-axis and y-axis
data = go.Scatter(x=scale, y=sim)

# Integrate the graphs' data
data1 = [data]

# create plots
py.offline.plot(data1, filename='Similarity.html')

print(min(sim))