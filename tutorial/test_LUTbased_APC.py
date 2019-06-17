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


'''Function version v9.0'''
def SumUpAPCLUT_dev(x):
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
def SumUpAPCLUT(x):
    # Save the input in the buffer
    t1 = copy.deepcopy(x)
    t2 = copy.deepcopy(x)

    # The shape of the input x: (sizeTensor+sizeBias, snLength)
    size, snLength = x.shape

    # Find the required number of APCs
    numAPC25 = int(size / 25)
    numAPC16 = int((size % 25) / 16)
    numAPC8 = int(((size % 25) % 16) / 8)

    # Initialize the variable
    sum25 = np.full(snLength, 0)
    sum16 = np.full(snLength, 0)
    sum8 = np.full(snLength, 0)

    if (numAPC25 != 0):
        # Remove the parts which are out of 25bit range
        x = x[:25 * numAPC25, :]

        # Transpose the input x: (snLength, sizeTensor+sizeBias)
        x = x.transpose()

        # Insert 7bit-zeros at every 25bits
        for i in range(numAPC25):
            for j in range(7):
                x = np.insert(x, i*32, False, axis=1)

        # Reshape it in order to pack in 16bits (4 x 8bits)
        x = x.reshape(snLength, -1, 4, 8)[:, :, ::-1]

        # Save the dimension information
        _, b, _, _ = x.shape

        # Pack the bits
        x = np.packbits(x).view(np.uint32)

        # Reshape it in order to handle the multiple APCs
        x = x.reshape(b, -1, order='F')

        # Look up the table
        for j in range(snLength):
            # Set the count number as 0
            jthSum = 0
            for i in range(numAPC25):
                jthSum += snLookupTableNumAPC25[x[i, j]]
            sum25[j] = jthSum

    if (numAPC16 != 0):
        t1 = t1[25 * numAPC25:(25 * numAPC25 + 16 * numAPC16), :]
        t1 = t1.transpose()
        t1 = t1.reshape(snLength, -1, 2, 8)
        _, b, _, _ = t1.shape
        t1 = np.packbits(t1).view(np.uint16)
        t1 = t1.reshape(b, -1, order='F')
        for j in range(snLength):
            jthSum = 0
            for i in range(numAPC16):
                jthSum += snLookupTableNumAPC16[t1[i, j]]
            sum16[j] = jthSum

    if (numAPC8 != 0):
        t2 = t2[(25 * numAPC25 + 16 * numAPC16):(25 * numAPC25 + 16 * numAPC16 + 8 * numAPC8), :]
        t2 = t2.transpose()
        t2 = t2.reshape(snLength, -1, 1, 8)
        _, b, _, _ = t2.shape
        t2 = np.packbits(t2).view(np.uint8)
        t2 = t2.reshape(b, -1, order='F')
        for j in range(snLength):
            jthSum = 0
            for i in range(numAPC8):
                jthSum += snLookupTableNumAPC8[t2[i, j]]
            sum8[j] = jthSum

    sum = sum25 + sum16 + sum8

    return sum


'''
def SumUpAPC25_dev(x, snLength, sizeTensor):
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
            ii = NOT(x[16 + 25 * i, j] & x[17 + 25 * i, j])
            jj = NOT(x[18 + 25 * i, j] | x[19 + 25 * i, j])
            k = NOT(x[20 + 25 * i, j] & x[21 + 25 * i, j])
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
            jthSum = jthSum + 16 * t4 + 8 * t3 + 4 * t2 + 2 * t1 + 1 * t0
            jthSum = 30 - jthSum

        sum[j] = jthSum

    return sum
'''

'''Function version v10.0'''
def Count2Integer(x, snLength, numAPC25, numAPC16, numAPC8):
    sumTotal = 0

    for i in range(len(x)):
        sumTotal = sumTotal + x[i]

    ret = (sumTotal / snLength) * 2 - (25 * numAPC25) - (16 * numAPC16) - (8 * numAPC8)
    return ret


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


# Load the lookup table for 8bit, 16bit and 25bit APC
snLookupTableNumAPC8 = 0
snLookupTableNumAPC16 = 0
snLookupTableNumAPC16Inversed = 0
snLookupTableNumAPC25 = 0
with open('snLookupTableNumAPC.pkl', 'rb') as input:
    snLookupTableNumAPC8 = pickle.load(input)
    snLookupTableNumAPC16 = pickle.load(input)
    _ = pickle.load(input)
    snLookupTableNumAPC25 = pickle.load(input)

# Initialize the graphs' data and parameters
SN_length = 2048
numSamples = 1000
numBitstreams = 25 + 16 + 8
result0 = np.zeros(numSamples)
reference4 = np.zeros(numSamples)
outputAPCInteger = np.zeros(numSamples)
count = np.full(SN_length, 0)
numAPC25 = int(numBitstreams / 25)
numAPC16 = int((numBitstreams % 25) / 16)
numAPC8 = int(((numBitstreams % 25) % 16) / 8)

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

# Add numbers using APCs
for i in range(values.shape[0]):
    count = SumUpAPCLUT(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length])
    # count = SumUpAPC25(partialSNs[(i*numBitstreams):((i+1)*numBitstreams), 0:SN_length],
    #                    SN_length,
    #                    numBitstreams)
    t = Count2Integer(count, SN_length, numAPC25, numAPC16, numAPC8)
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
