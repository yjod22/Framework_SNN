#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     test_SC-basedReLU.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   test_LUTbased_APC.py
# Version:     12.0
# Author/Date: Junseok Oh / 2019-06-25
# Change:      (SCR_V11.0-8): Refer to the class in holayer.py
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   test_LUTbased_APC.py
# Version:     10.0
# Author/Date: Junseok Oh / 2019-06-14
# Change:      (SCR_V9.0-1): Deploy SC-based ReLU
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   stanh_testing.py
# Version:     8.0
# Author/Date: Junseok Oh / 2019-05-23
# Change:      (SCR_V6.4-3): Update with tanh(2,3,4,5,6x)
#              (SCR_V6.4-9): Update Stanh with LUT for adaptive function
#              (SCR_V6.4-15): SC-Based Relu verification (failed)
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
import copy
import pickle
from snn.holayer import HOActivation

def createSN(x, length):
    """create bipolar SN by comparing random vector elementwise to SN value x"""
    large = np.random.rand(1)
    x_SN = np.full(length, False)
    if large:
        for i in range(int(np.ceil(((x+1)/2)*length))):
            x_SN[i] = True
    else:
        for i in range(int(np.floor(((x+1)/2)*length))):
            x_SN[i] = True
    np.random.shuffle(x_SN)
    return x_SN


def stochtoint(x):
    """convert bipolar stochastic number to integer"""
    return (sum(x)/len(x))*2 - 1


# Initialize the graphs' data and parameters
SN_length = 2048
numSamples = 1000
numBitstreams =32
# numStates = numBitstreams*4
result = np.zeros(numSamples)
reference = np.zeros(numSamples)
reference0 = np.zeros(numSamples)
reference1 = np.zeros(numSamples)
reference2 = np.zeros(numSamples)
reference3 = np.zeros(numSamples)
outputAPC = np.full((numSamples, SN_length), False)
count = np.full(SN_length, 0)

# get some random input values and sort them. These are going to be assigned to the x-axis
values = np.zeros(numSamples)
values = np.random.random(numSamples)*2*2 - 1*2
values = np.sort(values, 0)
partialValues = np.zeros(numSamples*numBitstreams)
for i in range(numSamples*numBitstreams):
    partialValues[i] = values[int(i/numBitstreams)] / numBitstreams

# produce the SNs for input values
#SNs = np.full((numSamples, SN_length), False)
#for i in range(values.shape[0]):
#    SNs[i] = createSN(values[i], SN_length)

partialSNs = np.full((numSamples*numBitstreams, SN_length), False)
for i in range(partialValues.shape[0]):
    partialSNs[i] = createSN(partialValues[i], SN_length)

# Refer to the class
hoActivation = HOActivation(activationFunc="default")

# apply stochastic function
for i in range(values.shape[0]):
    count, sizePreprocessed, numAPC25, numAPC16, numAPC8 = hoActivation.SumUpAPCLUT(partialSNs[(i*numBitstreams):((i+1)*numBitstreams),
                                                                                    0:SN_length])
    outputAPC[i] = hoActivation.UpDownCounterReLU(count, numBitstreams+sizePreprocessed, (numBitstreams+sizePreprocessed)*4)

# Calculate the graphs' data that are going to be assigned to the y-axis
for i in range(values.shape[0]):
    result[i] = stochtoint(outputAPC[i])
    reference[i] = min(max(0, values[i]), 1) # Clipped-ReLU
    reference0[i] = np.tanh(values[i]*1.84)
    reference1[i] = np.tanh(values[i]*1.29)
    reference2[i] = np.tanh(values[i])
    reference3[i] = np.tanh(values[i]*0.7)

# Assign the graphs' data to x-axis and y-axis
SCReLU = go.Scatter(x=values, y=result, mode='markers', name='SCReLU')
ReLU = go.Scatter(x=values, y=reference, name='ReLU')
Tanh1_84 = go.Scatter(x=values, y=reference0, name='Tanh(1.84*x)')
Tanh1_29 = go.Scatter(x=values, y=reference1, name='Tanh(1.29*x)')
Tanh = go.Scatter(x=values, y=reference2, name='Tanh(x)')
Tanh0_7 = go.Scatter(x=values, y=reference3, name='Tanh(0.7*x)')

# Integrate the graphs' data
data = [SCReLU, ReLU, Tanh1_84, Tanh1_29, Tanh, Tanh0_7]

# create plots
py.offline.plot(data, filename='testplot_ReLU.html')