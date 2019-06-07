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
import copy

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


'''GenerateLookupTableForSTanh, Function version v8.0'''
def GenerateLookupTableForSTanh(byte, start_state, PAR_numState):
    # Set the number of states
    #numState = 32
    numState = PAR_numState

    # Represent the decimal value into 8bit binary value
    x = np.unpackbits(np.array([byte], dtype='uint8'))

    # Set state to start_state
    state = start_state

    for j, bit in enumerate(x):
        # Determine the output depending on the current state
        x[j] = state > (numState/2 -1)

        # input is True -> increase state
        if state < (numState-1) and bit:
            state += 1
        # input is False -> decrease state
        if state > 0 and not bit:
            state -= 1

    return (np.packbits(x)[0], state)


'''ActivationFuncSTanhLUTSN, Function version v9.0'''
def ActivationFuncSTanhLUTSN(x, PAR_numState):
    # Represent the binary value into the 1byte decimal value
    sn = np.packbits(x)
    out = np.empty_like(sn)

    # Set the number of states
    numState = PAR_numState * 2
    state = max(int(numState/2) - 1, 0)

    _snLookupTableOut = snLookupTableOut[PAR_numState-1]
    _snLookupTableState = snLookupTableState[PAR_numState-1]

    for i, byte in enumerate(sn):
        # Find the element with the current state and byte information in the table
        # Fill in the out value with that element
        out[i] = _snLookupTableOut[state, byte]
        # Update the state according to the element in the table
        state = _snLookupTableState[state, byte]

    # Represent the decimal value into 8bit binary value
    out = np.unpackbits(out)
    return out


'''ActiavtionFuncTanhSN, Function version v9.0'''
def ActiavtionFuncTanhSN(x, PAR_numState):
    """activation function for stochastic NN"""
    # the number of states
    numState = PAR_numState*2
    # starting state
    state = max(int(numState/2)-1, 0)
    out = np.full(len(x), True, dtype=bool)
    for j in range(len(x)):
        # Determine the output depending on the current state
        if ((0 <= state) & (state < int(numState/2))):
            out[j] = 0
        else:
            out[j] = 1
        # input is True -> increase state
        if x[j] & (state < (numState-1)):
            state = state + 1
        # input is False -> decrease state
        elif np.logical_not(x[j]) & (state > 0):
            state = state - 1
        # No update at the start or end of the state
    return out


'''Function version v8.0'''
# Generate the lookup table for STanh activation function
# the number of scale factors is 20. i.e. tanh(1*x), tanh(2*x), ... , tanh(20*x)
# the number of states of STanh's state machine is determined by (num+1)*2
# 8bit [11111111] is equal to 255 in decimal
numScaleFactor = 20
snLookupTableOut = [ []for num in range(numScaleFactor)]
snLookupTableState = [ []for num in range(numScaleFactor)]
for num in range(numScaleFactor):
    snLookupTableElementsTemp = np.array(
        #[[self.GenerateLookupTableForSTanh(byte, state) for byte in range(256)] for state in range(32)])
        [[GenerateLookupTableForSTanh(byte, state, (num+1)*2) for byte in range(256)] for state in range((num+1)*2)])
    snLookupTableOut[num] = copy.deepcopy(snLookupTableElementsTemp[:, :, 0])
    snLookupTableState[num] = copy.deepcopy(snLookupTableElementsTemp[:, :, 1])
del(snLookupTableElementsTemp)


# Initialize the graphs' data and parameters
SN_length = 2048
numSamples = 1000
result1 = np.zeros(numSamples)
result2 = np.zeros(numSamples)
result3 = np.zeros(numSamples)
result4 = np.zeros(numSamples)
result5 = np.zeros(numSamples)
result6 = np.zeros(numSamples)
result7 = np.zeros(numSamples)
reference1 = np.zeros(numSamples)
reference2 = np.zeros(numSamples)
reference3 = np.zeros(numSamples)
reference4 = np.zeros(numSamples)
reference5 = np.zeros(numSamples)
reference6 = np.zeros(numSamples)
reference7 = np.zeros(numSamples)


# get some random input values and sort them. These are going to be assigned to the x-axis
values = np.zeros(numSamples)
values = np.random.random(numSamples)*2 - 1
values = np.sort(values, 0)


# produce the SNs for input values
SNs = np.full((numSamples, SN_length), False)
for i in range(values.shape[0]):
    SNs[i] = createSN(values[i], SN_length)


# Calculate the graphs' data that are going to be assigned to the y-axis
for i in range(values.shape[0]):
    result1[i] = stochtoint(ActivationFuncSTanhLUTSN(SNs[i], 1))
    result2[i] = stochtoint(ActivationFuncSTanhLUTSN(SNs[i], 2))
    result3[i] = stochtoint(ActivationFuncSTanhLUTSN(SNs[i], 3))
    result4[i] = stochtoint(ActivationFuncSTanhLUTSN(SNs[i], 4))
    result5[i] = stochtoint(ActivationFuncSTanhLUTSN(SNs[i], 5))
    result6[i] = stochtoint(ActivationFuncSTanhLUTSN(SNs[i], 6))
    result7[i] = stochtoint(ActivationFuncSTanhLUTSN(SNs[i], 7))
    reference1[i] = np.tanh(values[i]*1)
    reference2[i] = np.tanh(values[i]*2)
    reference3[i] = np.tanh(values[i]*3)
    reference4[i] = np.tanh(values[i]*4)
    reference5[i] = np.tanh(values[i]*5)
    reference6[i] = np.tanh(values[i]*6)
    reference7[i] = np.tanh(values[i]*7)


# Assign the graphs' data to x-axis and y-axis
stanh1x = go.Scatter(x=values, y=result1)
tanh1x = go.Scatter(x=values, y=reference1)

stanh2x = go.Scatter(x=values, y=result2)
tanh2x = go.Scatter(x=values, y=reference2)

stanh3x = go.Scatter(x=values, y=result3)
tanh3x = go.Scatter(x=values, y=reference3)

stanh4x = go.Scatter(x=values, y=result4)
tanh4x = go.Scatter(x=values, y=reference4)

stanh5x = go.Scatter(x=values, y=result5)
tanh5x = go.Scatter(x=values, y=reference5)

stanh6x = go.Scatter(x=values, y=result6)
tanh6x = go.Scatter(x=values, y=reference6)

stanh7x = go.Scatter(x=values, y=result7)
tanh7x = go.Scatter(x=values, y=reference7)


# Integrate the graphs' data
data1 = [stanh1x, tanh1x]
data2 = [stanh2x, tanh2x]
data3 = [stanh3x, tanh3x]
data4 = [stanh4x, tanh4x]
data5 = [stanh5x, tanh5x]
data6 = [stanh6x, tanh6x]
data7 = [stanh7x, tanh7x]


# create plots
py.offline.plot(data1, filename='STanh(1x).html')
py.offline.plot(data2, filename='STanh(2x).html')
py.offline.plot(data3, filename='STanh(3x).html')
py.offline.plot(data4, filename='STanh(4x).html')
py.offline.plot(data5, filename='STanh(5x).html')
py.offline.plot(data6, filename='STanh(6x).html')
py.offline.plot(data7, filename='STanh(7x).html')