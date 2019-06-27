#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     stanh_testing.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
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

def SumUpAPC(x, snLength, sizeTensor):
    # sizeState = r
    # snLength = m
    # sizeTensor = n
    numAPC = int(sizeTensor / 16)
    sum = np.full(snLength, 0)

    for j in range(snLength):
        # Set the count number as 0
        jthSum = 0

        # Count the number of 1s on each column approximately
        # and save the result in jthSum
        for i in range(numAPC):
            # AND, OR gates
            a = (x[0 + 16 * i, j] | x[1 + 16 * i, j])
            b = (x[2 + 16 * i, j] & x[3 + 16 * i, j])
            c = (x[4 + 16 * i, j] | x[5 + 16 * i, j])
            d = (x[6 + 16 * i, j] & x[7 + 16 * i, j])
            e = (x[8 + 16 * i, j] | x[9 + 16 * i, j])
            f = (x[10 + 16 * i, j] & x[11 + 16 * i, j])
            z2 = (x[12 + 16 * i, j] | x[13 + 16 * i, j])
            t0 = (x[14 + 16 * i, j] & x[15 + 16 * i, j])

            # Full Adder 1 (Carry:x1, Sum:x2)
            x1 = ((a & b) | (b & c) | (c & a))
            x2 = ((a ^ b) ^ c)

            # Full Adder 2 (Carry:y1, Sum:y2)
            y1 = ((d & e) | (e & f) | (f & d))
            y2 = ((d ^ e) ^ f)

            # Full Adder 3 (Carry:z1, Sum:t1)
            z1 = ((x2 & y2) | (y2 & z2) | (z2 & x2))
            t1 = ((x2 ^ y2) ^ z2)

            # Full Adder 4 (Carry:t3, Sum:t2)
            t3 = ((x1 & y1) | (y1 & z1) | (z1 & x1))
            t2 = ((x1 ^ y1) ^ z1)

            # Represent in the binary format
            jthSum = jthSum + 8 * t3 + 4 * t2 + 2 * t1 + 2 * t0

        sum[j] = jthSum

    return sum

def Count2Integer(x, snLength, numAPC):
    sumTotal = 0

    for i in range(len(x)):
        sumTotal = sumTotal + x[i]

    ret = (sumTotal / snLength) * 2 - (16 * numAPC)
    return ret

def UpDownCounterWithAccumulator(x, sizeTensor, sizeState):
    # It works as the SC-Based ReLU
    # the parameter sizeTensor here refers to (sizeTensor+sizeBias)

    # len(x) = m
    # sizeTensor = n
    # sizeState = r
    stateMax = sizeState - 1
    stateHalf = int(sizeState / 2)
    stateCurrent = stateHalf
    y = np.full(len(x), False)
    cntAccumulated = 0

    for i in range(len(x)):
        # Bipolar 1s counting
        v = x[i] * 2 - sizeTensor

        # Update the current state
        stateCurrent = stateCurrent + v

        # Accumulate current column of the input
        cntAccumulated += x[i]

        # Enforce the output of ReLU to be greater than or equal to 0
        cycleHalf = int((i+1)*(sizeTensor/2))
        if(cntAccumulated < cycleHalf):
            y[i] = 1
            cntAccumulated += 1

        # Otherwise the output is determined by the following FSM
        else:
            # Check the exceptional cases
            if (stateCurrent > stateMax):
                stateCurrent = stateMax
            if (stateCurrent < 0):
                stateCurrent = 0

            # Generate the output using the stochastic number
            if (stateCurrent > stateHalf):
                y[i] = 1
            else:
                y[i] = 0
    return y

def SCBasedRelu(x, sizeState):
    # It works as the SC-Based ReLU
    # the parameter sizeTensor here refers to (sizeTensor+sizeBias)

    # len(x) = m
    # sizeTensor = n
    # sizeState = r
    stateMax = sizeState - 1
    stateHalf = int(sizeState / 2)
    stateCurrent = stateHalf
    y = np.full(len(x), False)
    cntAccumulated = 0

    for i in range(len(x)):
        # Bipolar 1s counting
        v = x[i] * 2 - 1

        # Update the current state
        stateCurrent = stateCurrent + v

        # Accumulate current column of the input
        cntAccumulated += x[i]

        # Enforce the output of ReLU to be greater than or equal to 0
        cycleHalf = int(i+1)
        if(cntAccumulated < cycleHalf):
            y[i] = 1
            cntAccumulated += 1

        # Otherwise the output is determined by the following FSM
        else:
            # Check the exceptional cases
            if (stateCurrent > stateMax):
                stateCurrent = stateMax
            if (stateCurrent < 0):
                stateCurrent = 0

            # Generate the output using the stochastic number
            if (stateCurrent > stateHalf):
                y[i] = 1
            else:
                y[i] = 0
    return y

def UpDownCounter(x, sizeTensor, sizeState):
    #len(x) = m
    #sizeTensor = n
    #sizeState = r
    stateMax = sizeState - 1
    stateHalf = int(sizeState / 2)
    stateCurrent = stateHalf
    y = np.full(len(x), False)

    for i in range(len(x)):
        v = x[i] * 2 - sizeTensor

        # Update the current state
        stateCurrent = stateCurrent + v

        # Check the exceptional cases
        if (stateCurrent > stateMax):
            stateCurrent = stateMax
        if (stateCurrent < 0):
            stateCurrent = 0

        # Generate the output using the stochastic number
        if (stateCurrent > stateHalf):
            y[i] = 1
        else:
            y[i] = 0

    return y

### function needed

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

# Generate the lookup table for STanh activation function
# the number of scale factors is 20. i.e. tanh(1*x), tanh(2*x), ... , tanh(20*x)
# the number of states of STanh's state machine is determined by (num+1)*2
# 8bit [11111111] is equal to 255 in decimal
numScaleFactor = 2
snLookupTableOut = [ []for num in range(numScaleFactor)]
snLookupTableState = [ []for num in range(numScaleFactor)]
for num in range(numScaleFactor):
    snLookupTableElementsTemp = np.array(
        #[[self.GenerateLookupTableForSTanh(byte, state) for byte in range(256)] for state in range(32)])
        [[GenerateLookupTableForSTanh(byte, state, (num+1)*2) for byte in range(256)] for state in range((num+1)*2)])
    snLookupTableOut[num] = copy.deepcopy(snLookupTableElementsTemp[:, :, 0])
    snLookupTableState[num] = copy.deepcopy(snLookupTableElementsTemp[:, :, 1])
del(snLookupTableElementsTemp)

def ActivationFuncSTanhLUTSN(x, PAR_numState):
    # Represent the binary value into the 1byte decimal value
    sn = np.packbits(x)
    out = np.empty_like(sn)

    # Set the number of states
    numState = PAR_numState * 2
    state = int(numState/2) - 1

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

def stanh2(x):
    """activation function for stochastic NN. FSN with 4 states"""
    # starting state
    state = 2
    out = np.full(len(x), True, dtype=bool)
    for j in range(len(x)):
        # input is True -> increase state
        if x[j] & (state < 4):
            if state < 2:
                out[j] = 0
            else:
                out[j] = 1
            state = state + 1
        elif x[j] & (state == 4):
            out[j] = 1
        # input is False -> decrease state
        elif (np.logical_not(x[j])) & (state > 0):
            if state > 2:
                out[j] = 1
            else:
                out[j] = 0
            state = state - 1
        elif (np.logical_not(x[j])) & (state == 0):
            out[j] = 0
    return out

def stanh(x):
    """activation function for stochastic NN. FSN with 14 states"""
    # starting state
    state = 7
    out = np.full(len(x), True, dtype=bool)
    for j in range(len(x)):
        # input is True -> increase state
        if x[j] & (state < 13):
            if state < 7:
                out[j] = 0
            else:
                out[j] = 1
            state = state + 1
        elif x[j] & (state == 13):
            out[j] = 1
        # input is False -> decrease state
        elif (np.logical_not(x[j])) & (state > 0):
            if state > 7:
                out[j] = 1
            else:
                out[j] = 0
            state = state - 1
        elif (np.logical_not(x[j])) & (state == 0):
            out[j] = 0
    return out


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


def createSN2(x, length):
    rand = np.random.rand(length)*2.0 - 1.0
    x_SN = np.less(rand, x)
    return x_SN


def stochtoint(x):
    """convert bipolar stochastic number to integer"""
    return (sum(x)/len(x))*2 - 1


SN_length = 2048
numSamples = 1000
numBitstreams =32
numStates = 64*2

# get some random input values and sort them
values = np.zeros(numSamples)
values = np.random.random(numSamples)*2 - 1
values = np.sort(values, 0)

partialValues = np.zeros(numSamples*numBitstreams)
for i in range(numSamples*numBitstreams):
    partialValues[i] = values[int(i/numBitstreams)] / numBitstreams

# produce the SNs for input values
SNs = np.full((numSamples, SN_length), False)
for i in range(values.shape[0]):
    SNs[i] = createSN(values[i], SN_length)

partialSNs = np.full((numSamples*numBitstreams, SN_length), False)
for i in range(partialValues.shape[0]):
    partialSNs[i] = createSN(partialValues[i], SN_length)

# apply stochastic tanh function
output = np.full((numSamples, SN_length), False)
output2 = np.full((numSamples, SN_length), False)
output3 = np.full((numSamples, SN_length), False)
output4 = np.full((numSamples, SN_length), False)
output5 = np.full((numSamples, SN_length), False)
output6 = np.full((numSamples, SN_length), False)
outputAPC = np.full((numSamples, SN_length), False)
outputRelu = np.full((numSamples, SN_length), False)
outputAPCInteger = np.zeros(numSamples)
count = np.full(SN_length, 0)
numAPC = int(numBitstreams / 16)

for i in range(values.shape[0]):
    output[i] = stanh(SNs[i])
    #output2[i] = ActiavtionFuncTanhSN(SNs[i], 2)
    #output2[i] = stanh2(SNs[i])
    #output3[i] = ActivationFuncSTanhLUTSN(SNs[i], 3)
    #output4[i] = ActivationFuncSTanhLUTSN(SNs[i], 4)
    #output5[i] = ActivationFuncSTanhLUTSN(SNs[i], 5)
    #output6[i] = ActivationFuncSTanhLUTSN(SNs[i], 6)

    count = SumUpAPC(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length],
                     SN_length,
                     numBitstreams)
    outputAPC[i] = UpDownCounter(count, numBitstreams, numStates)

    outputRelu[i] = SCBasedRelu(outputAPC[i], 4)
    #outputAPC[i] = UpDownCounterWithAccumulator(count, numBitstreams, numStates)

    #t = np.tanh(1 * Count2Integer(count, SN_length, numAPC))
    #outputAPCInteger[i] = t

result1 = np.zeros(numSamples)
result2 = np.zeros(numSamples)
#result3 = np.zeros(numSamples)
result4 = np.zeros(numSamples)
result5 = np.zeros(numSamples)
result6 = np.zeros(numSamples)
result7 = np.zeros(numSamples)
result8 = np.zeros(numSamples)
reference1 = np.zeros(numSamples)
reference2 = np.zeros(numSamples)
#reference3 = np.zeros(numSamples)
reference4 = np.zeros(numSamples)
reference5 = np.zeros(numSamples)
reference6 = np.zeros(numSamples)
reference7 = np.zeros(numSamples)
reference8 = np.zeros(numSamples)

# calculate results and reference results
for i in range(values.shape[0]):
    result1[i] = stochtoint(output[i])
    #result2[i] = stochtoint(outputAPC[i])
    result2[i] = stochtoint(outputRelu[i])
    #result3[i] = outputAPCInteger[i]
    #result4[i] = stochtoint(output2[i])
    #result5[i] = stochtoint(output3[i])
    #result6[i] = stochtoint(output4[i])
    #result7[i] = stochtoint(output5[i])
    #result8[i] = stochtoint(output6[i])
    reference1[i] = np.tanh(values[i]*7)
    #reference2[i] = np.tanh(values[i]*0.7)
    reference2[i] = min(max(0, values[i]), 1)
    #reference3[i] = np.tanh(values[i]*1)
    #reference4[i] = np.tanh(values[i]*2)
    #reference5[i] = np.tanh(values[i]*3)
    #reference6[i] = np.tanh(values[i]*4)
    #reference7[i] = np.tanh(values[i]*5)
    #reference8[i] = np.tanh(values[i]*6)

# create plots
stanhx = go.Scatter(x = values, y = result1)
APC_btanhx = go.Scatter(x=values, y=result2)
#APC_tanhx = go.Scatter(x=values, y=result3)
#stanh2x = go.Scatter(x=values, y=result4)
#stanh3x = go.Scatter(x=values, y=result5)
#stanh4x = go.Scatter(x=values, y=result6)
#stanh5x = go.Scatter(x=values, y=result7)
#stanh6x = go.Scatter(x=values, y=result8)
tanh7x = go.Scatter(x=values, y=reference1)
tanh0_7x = go.Scatter(x=values, y=reference2)
#tanhx = go.Scatter(x=values, y=reference3)
#tanh2x = go.Scatter(x=values, y=reference4)
#tanh3x = go.Scatter(x=values, y=reference5)
#tanh4x = go.Scatter(x=values, y=reference6)
#tanh5x = go.Scatter(x=values, y=reference7)
#tanh6x = go.Scatter(x=values, y=reference8)

#data1 = [stanhx, tanh7x, APC_btanhx, tanhx, APC_tanhx, tanhx]
#data1 = [stanhx, tanh7x, stanh2x, tanh2x, stanh3x, tanh3x, stanh4x, tanh4x, stanh5x, tanh5x, stanh6x, tanh6x]
data2 = [stanhx, tanh7x]
data3 = [APC_btanhx, tanh0_7x]
#data4 = [APC_tanhx, tanhx]
#data5 = [stanh2x, tanh2x]
#data6 = [stanh3x, tanh3x]
#data7 = [stanh4x, tanh4x]
#data8 = [stanh5x, tanh5x]
#data9 = [stanh6x, tanh6x]

#py.offline.plot(data1, filename='testplot1.html')
py.offline.plot(data2, filename='testplot2.html')
py.offline.plot(data3, filename='testplot3.html')
#py.offline.plot(data4, filename='testplot4.html')
#py.offline.plot(data5, filename='testplot5.html')
#py.offline.plot(data6, filename='testplot6.html')
#py.offline.plot(data7, filename='testplot7.html')
#py.offline.plot(data8, filename='testplot8.html')
#py.offline.plot(data9, filename='testplot9.html')
