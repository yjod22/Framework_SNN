#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     test_SC-based_Tanh.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   test_SC-based_Tanh.py
# Version:     12.0
# Author/Date: Junseok Oh / 2019-06-25
# Change:      (SCR_V11.0-3): Define SumUpPC, SumUpAPC8,16,25 for verifying (A)PC+BTanh
#              (SCR_V11.0-8): Refer to the class in holayer.py
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

'''Function version v12.0'''
def SumUpPC(x):
    sizeInput, snLength = x.shape
    t = np.full(snLength, 0)
    x = x.transpose()
    for i in range (snLength):
        t[i] = sum(x[i])
    return t


'''Function version v12.0'''
def SumUpAPC8(x, snLength, sizeTensor):
    # sizeState = r
    # snLength = m
    # sizeTensor = n
    numAPC = int(sizeTensor / 8)
    sum = np.full(snLength, 0)

    for j in range(snLength):
        # Set the count number as 0
        jthSum = 0

        # Count the number of 1s on each column approximately
        # and save the result in jthSum
        for i in range(numAPC):
            # AND, OR gates
            a = (x[0 + 8 * i, j] | x[1 + 8 * i, j])
            b = (x[2 + 8 * i, j] & x[3 + 8 * i, j])
            c = (x[4 + 8 * i, j] | x[5 + 8 * i, j])
            t0 = (x[6 + 8 * i, j] & x[7 + 8 * i, j])

            # Full Adder 1 (Carry:x1, Sum:x2)
            t2 = ((a & b) | (b & c) | (c & a))
            t1 = ((a ^ b) ^ c)

            # Represent in the binary format
            jthSum = jthSum + 4 * t2 + 2 * t1 + 2 * t0

        sum[j] = jthSum

    return sum


'''Function version v12.0'''
def SumUpAPC16(x, snLength, sizeTensor):
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


'''Function version v10.0'''
def NOT(a):
    if(a == 0):
        return 1
    elif(a == 1):
        return 0


'''Function version v12.0'''
def SumUpAPC25(x, snLength, sizeTensor):
    # sizeState = r
    # snLength = m
    # sizeTensor = n
    numAPC = int(sizeTensor / 25)
    sum = np.full(snLength, 0)

    for j in range(snLength):
        # Set the count number as 0
        jthSum = 0

        # Count the number of 1s on each column approximately
        # and save the result in jthSum
        for i in range(numAPC):
            # NAND, NOR gates
            a = NOT(x[0 + 25 * i, j] | x[1 + 25 * i, j])
            b = NOT(x[2 + 25 * i, j] & x[3 + 25 * i, j])
            c = NOT(x[4 + 25 * i, j] | x[5 + 25 * i, j])
            d = NOT(x[6 + 25 * i, j] & x[7 + 25 * i, j])
            e = NOT(x[8 + 25 * i, j] | x[9 + 25 * i, j])
            f = NOT(x[10 + 25 * i, j] & x[11 + 25 * i, j])
            g = NOT(x[12 + 25 * i, j] | x[13 + 25 * i, j])
            h = NOT(x[14 + 25 * i, j] & x[15 + 25 * i, j])
            ii = NOT(x[16 + 25 * i, j] | x[17 + 25 * i, j])
            jj = NOT(x[18 + 25 * i, j] & x[19 + 25 * i, j])
            k = NOT(x[20 + 25 * i, j] | x[21 + 25 * i, j])
            l = NOT(x[22 + 25 * i, j] & x[23 + 25 * i, j])
            t0 = NOT(x[24 + 25 * i, j])

            # Inversed Full Adder 1 (Carry:m, Sum:n)
            m = NOT((a & b) | (b & c) | (c & a))
            n = NOT((a ^ b) ^ c)

            # Inversed Full Adder 2 (Carry:o, Sum:p)
            o = NOT((d & e) | (e & f) | (f & d))
            p = NOT((d ^ e) ^ f)

            # Inversed Full Adder 3 (Carry:q, Sum:r)
            q = NOT((g & h) | (h & ii) | (ii & g))
            r = NOT((g ^ h) ^ ii)

            # Inversed Full Adder 4 (Carry:s, Sum:t)
            s = NOT((jj & k) | (k & l) | (l & jj))
            t = NOT((jj ^ k) ^ l)

            # Inversed Half Adder 1 (Carry:w, Sum:x)
            w = (n & p)
            xx = NOT(n ^ p)

            # Inversed Half Adder 2 (Carry:A, Sum:B)
            A = (r & t)
            B = NOT(r ^ t)

            # Inversed Full Adder 5 (Carry:u, Sum:v)
            u = NOT((m & o) | (o & w) | (w & m))
            v = NOT((m ^ o) ^ w)

            # Inversed Full Adder 6 (Carry:y, Sum:z)
            y = NOT((q & s) | (s & A) | (A & q))
            z = NOT((q ^ s) ^ A)

            # Half Adder 1 (Carry:D, Sum:t1)
            D = (xx & B)
            t1 = (xx ^ B)

            # Full Adder 1 (Carry:C, Sum:t2)
            C = ((v & z) | (z & D) | (D & v))
            t2 = ((v ^ z) ^ D)

            # Full Adder 2 (Carry:t4, Sum:t3)
            t4 = ((u & y) | (y & C) | (C & u))
            t3 = ((u ^ y) ^ C)

            # Represent in the binary format
            partialSum = 16 * t4 + 8 * t3 + 4 * t2 + 2 * t1 + 1 * t0
            partialSum = 29 - partialSum
            jthSum = jthSum + partialSum

        sum[j] = jthSum

    return sum


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
# numBitstreams =25*4*10
numBitstreams =25
# numStates = numBitstreams*4
numStates = 94
result = np.zeros(numSamples)
reference = np.zeros(numSamples)
reference0 = np.zeros(numSamples)
reference1 = np.zeros(numSamples)
outputAPC = np.full((numSamples, SN_length), False)
count = np.full(SN_length, 0)

# get some random input values and sort them. These are going to be assigned to the x-axis
values = np.zeros(numSamples)
values = np.random.random(numSamples)*2*5 - 1*5
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
    count, _, numAPC25, numAPC16, numAPC8 = hoActivation.SumUpAPCLUT(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length])
    # count = SumUpPC(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length])
    # count = SumUpAPC8(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length],
    #                  SN_length,
    #                  numBitstreams)
    # count = SumUpAPC16(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length],
    #                  SN_length,
    #                  numBitstreams)
    # count = SumUpAPC25(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length],
    #                    SN_length,
    #                    numBitstreams)
    outputAPC[i] = hoActivation.UpDownCounter(count, numBitstreams, numStates)

# Calculate the graphs' data that are going to be assigned to the y-axis
for i in range(values.shape[0]):
    result[i] = stochtoint(outputAPC[i])
    # reference[i] = min(max(0, values[i]), 1) # Clipped-ReLU
    reference0[i] = np.tanh(values[i]*1.9740*0.7)
    reference1[i] = np.tanh(values[i]*1.9740)
	
# Assign the graphs' data to x-axis and y-axis
BTanh = go.Scatter(x=values, y=result, mode='markers', name='BTanh')
# ReLU = go.Scatter(x=values, y=reference, name='ReLU')
Ref0 = go.Scatter(x=values, y=reference0, name='Tanh(1.3818*x)')  # For APC, 1.3818 = 0.7*1.9740
Ref1 = go.Scatter(x=values, y=reference1, name='Tanh(1.9740*x)')  # For PC

# Integrate the graphs' data
data = [BTanh, Ref0, Ref1]

# create plots
py.offline.plot(data, filename='testplot_BTanh.html')


