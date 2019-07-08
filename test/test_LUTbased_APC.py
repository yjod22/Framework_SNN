#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     test_LUTbased_APC.py
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
# Version:     15.0
# Author/Date: Junseok Oh / 2019-07-05
# Change:      (SCR_V14.0-1): Modularize the classes, change the file names
#              (SCR_V14.0-6): Update test files so that it is referred to the current SW
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   test_LUTbased_APC.py
# Version:     12.0
# Author/Date: Junseok Oh / 2019-06-25
# Change:      (SCR_V11.0-2): Define SumUpAPC25 for debugging purpose
#              (SCR_V11.0-8): Refer to the class in holayer.py
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   test_LUTbased_APC.py
# Version:     11.0
# Author/Date: Junseok Oh / 2019-06-18
# Change:      (SCR_V10.0-1): Pre-processing in APCs
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   test_LUTbased_APC.py
# Version:     10.0
# Author/Date: Junseok Oh / 2019-06-14
# Change:      (SCR_V9.0-2): Deploy APCs(8, 16, 25bits)
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   test_LUTbased_APC.py
# Version:     9.0
# Author/Date: Junseok Oh / 2019-06-07
# Change:      (SCR_V8.0-3): develop LUT-based APC
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
import pickle
from snn.hoLayer import HOActivation

'''
# Dimension altering practice
a = np.random.randint(0, 2, (4, 16))
t1 = a.reshape(-1, 2, 8)
t2 = a.reshape(-1, 2, 8)[:, ::-1]
b = np.packbits(a.reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)


a = np.random.randint(0, 2, (33, 10))
a = a[:16*2, :]
b = a.transpose()
# t1 = b.reshape(10, 2, 2, 8)
t1 = b.reshape(10, -1, 2, 8)
t2 = b.reshape(10, 2, 2, 8)[:, :, ::-1]
i, j, k, l = t2.shape
t3 = np.packbits(t2).view(np.uint16)
t4 = t3.reshape(2, -1, order='F')
'''

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


# Initialize the graphs' data and parameters
SN_length = 2048
numSamples = 1000
# numBitstreams = 5                 # 8bit-APC
# numBitstreams = 8                 # 8bit-APC
# numBitstreams = 15                # 16bit-APC
# numBitstreams = 16                # 16bit-APC
# numBitstreams = 24                # 25bit-APC
numBitstreams = 25                # 25bit-APC
# numBitstreams = 25 + 5            # 25bit-APC + 8bit-APC
# numBitstreams = 25 + 8            # 25bit-APC + 8bit-APC
# numBitstreams = 25 + 15           # 25bit-APC + 16bit-APC
# numBitstreams = 25 + 16           # 25bit-APC + 16bit-APC
# numBitstreams = 25 + 24           # 25bit-APC + 25bit-APC
# numBitstreams = 25 + 25           # 25bit-APC + 25bit-APC
# numBitstreams = 25 + 25 + 5       # 25bit-APC + 25bit-APC + 8bit-APC
# numBitstreams = 25 + 25 + 15      # 25bit-APC + 25bit-APC + 16bit-APC
# numBitstreams = 25 + 25 + 25      # 25bit-APC + 25bit-APC + 25bit-APC
result0 = np.zeros(numSamples)
reference4 = np.zeros(numSamples)
outputAPCInteger = np.zeros(numSamples)
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
hoActivation = HOActivation(kBits=11, baseMode="APC", activationFunc="default")

# Add numbers using APCs
for i in range(values.shape[0]):
    # count, _, numAPC25, numAPC16, numAPC8 = SumUpAPCLUT(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length])
    count, _, numAPC25, numAPC16, numAPC8 = hoActivation.SumUpAPCLUT(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length])
    # count = SumUpAPC25(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length],
    #                    SN_length,
    #                    numBitstreams)
    # numAPC25 = 1
    # numAPC16 = 0
    # numAPC8 = 0
    t = hoActivation.Count2Integer(count, SN_length, numAPC25, numAPC16, numAPC8)
    # t = np.tanh(1 * Count2Integer(count, SN_length, numAPC16, numAPC8))
    outputAPCInteger[i] = t

# Calculate the graphs' data that are going to be assigned to the y-axis
for i in range(values.shape[0]):
    result0[i] = outputAPCInteger[i]
    reference4[i] = values[i] # y = x

# Assign the graphs' data to x-axis and y-axis
APC_y_x = go.Scatter(x=values, y=result0, mode='markers', name='APC_y_x')
y_x = go.Scatter(x=values, y=reference4, name='y = x')

# Integrate the graphs' data
data = [APC_y_x, y_x]

# create plots
py.offline.plot(data, filename='testplot_LUTbased_APC.html')
