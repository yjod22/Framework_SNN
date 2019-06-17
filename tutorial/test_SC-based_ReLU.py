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


'''Function version v9.0'''
def GenerateLookupTableForAPC8(Byte):
    # Represent the decimal value into 8bit binary value
    x = np.unpackbits(np.array([Byte], dtype='uint8'))

    # Initialize the sum
    sum = 0

    # AND, OR gates
    a = (x[0] | x[1])
    b = (x[2] & x[3])
    c = (x[4] | x[5])
    t0 = (x[6] & x[7])

    # Full Adder 1 (Carry:x1, Sum:x2)
    t2 = ((a & b) | (b & c) | (c & a))
    t1 = ((a ^ b) ^ c)

    # Represent in the binary format
    sum = 4 * t2 + 2 * t1 + 2 * t0

    return sum


'''Function version v9.0'''
def unpack16bits(in_intAr, Nbits):
    ''' convert (numpyarray of uint => array of Nbits bits) for many bits in parallel'''
    inSize_T = in_intAr.shape
    in_intAr_flat = in_intAr.flatten()
    out_NbitAr = np.zeros((len(in_intAr_flat), Nbits))
    for iBits in range(Nbits):
        out_NbitAr[:, iBits] = (in_intAr_flat >> iBits) & 1
    out_NbitAr = out_NbitAr.reshape(inSize_T + (Nbits,))
    return out_NbitAr


'''Function version v9.0'''
def GenerateLookupTableForAPC16(twoByte):
    twoByte = np.array(twoByte)
    # Represent the decimal value into 16bit binary value
    x = unpack16bits(twoByte, 16).astype('uint16')
    x = x[::-1]

    # Initialize the sum
    sum = 0

    # AND, OR gates
    a = (x[0] | x[1])
    b = (x[2] & x[3])
    c = (x[4] | x[5])
    d = (x[6] & x[7])
    e = (x[8] | x[9])
    f = (x[10] & x[11])
    z2 = (x[12] | x[13])
    t0 = (x[14] & x[15])

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
    sum = 8 * t3 + 4 * t2 + 2 * t1 + 2 * t0

    return sum


'''Function version v9.0'''
def SumUpAPCLUT(x):
    # Save the input in the buffer
    t = copy.deepcopy(x)

    # The shape of the input x: (sizeTensor+sizeBias, snLength)
    size, snLength = x.shape

    # Find the required number of APC16 and APC8
    numAPC16 = int(size / 16)
    numAPC8 = int((size % 16) / 8)
    numAPC = numAPC8 + numAPC16

    # Initialize the variable
    sum16 = np.full(snLength, 0)
    sum8 = np.full(snLength, 0)
    sum = np.full(snLength, 0)

    # Remove the parts which are out of 16bit range
    x = x[:16 * numAPC16, :]

    # Transpose the input x: (snLength, sizeTensor+sizeBias)
    x = x.transpose()

    # Reshape it in order to pack in 16bits (2 x 8bits)
    x = x.reshape(snLength, -1, 2, 8)[:, :, ::-1]

    # Save the dimension information
    _, b, _, _ = x.shape

    # Pack the bits
    x = np.packbits(x).view(np.uint16)

    # Reshape it in order to handle the multiple APCs
    x = x.reshape(b, -1, order='F')

    # Look up the table
    for j in range(snLength):
        # Set the count number as 0
        jthSum = 0
        for i in range(numAPC16):
            jthSum += snLookupTableNumAPC16[x[i, j]]
        sum16[j] = jthSum

    if (numAPC8 != 0):
        t = t[16 * numAPC16:(16 * numAPC16 + 8 * numAPC8), :]
        t = t.transpose()
        t = t.reshape(snLength, -1, 1, 8)
        _, b, _, _ = t.shape
        t = np.packbits(t).view(np.uint8)
        t = t.reshape(b, -1, order='F')
        for j in range(snLength):
            jthSum = 0
            for i in range(numAPC8):
                jthSum += snLookupTableNumAPC8[t[i, j]]
            sum8[j] = jthSum

    sum = sum8 + sum16

    return sum


'''Function version v10.0'''
def UpDownCounterReLU(x, sizeTensor, sizeState):
    # the parameter sizeTensor here refers to (sizeTensor+sizeBias)

    # len(x) = m
    # sizeTensor = n
    # sizeState = r
    stateMax = sizeState - 1
    stateHalf = int(sizeState / 2)
    stateCurrent = stateHalf
    y = np.full(len(x), False)
    accumulated = 0

    for i in range(len(x)):
        # Bipolar 1s counting
        v = x[i] * 2 - sizeTensor

        # Update the current state
        stateCurrent = stateCurrent + v

        # Check the exceptional cases
        if (stateCurrent > stateMax):
            stateCurrent = stateMax
        if (stateCurrent < 0):
            stateCurrent = 0

        # Enforce the output of ReLU to be greater than or equal to 0
        if(accumulated < int(i/2)):
            y[i] = 1

        # Otherwise, the output is  determined by the following FSM
        else:
            # Generate the output using the stochastic number
            if (stateCurrent > stateHalf):
                y[i] = 1
            else:
                y[i] = 0

        # Accumulate the current output
        accumulated += y[i]

    return y

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


# Generate the lookup table for 16bit and 8bit APC
snLookupTableNumAPC16 = np.array([GenerateLookupTableForAPC16(twoByte) for twoByte in range(65536)])
snLookupTableNumAPC8 = np.array([GenerateLookupTableForAPC8(Byte) for Byte in range(256)])


# Initialize the graphs' data and parameters
SN_length = 2048
numSamples = 1000
numBitstreams =32
numStates = numBitstreams*4
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

# apply stochastic function
for i in range(values.shape[0]):
    count = SumUpAPCLUT(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length])
    outputAPC[i] = UpDownCounterReLU(count, numBitstreams, numStates)

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