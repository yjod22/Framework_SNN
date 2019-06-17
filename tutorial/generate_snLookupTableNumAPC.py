#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     generate_snLookupTableNumAPC.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   generate_snLookupTableNumAPC.py
# Version:     10.0
# Author/Date: Junseok Oh / 2019-06-16
# Change:      (SCR_V9.0-3): Generate snLookupTableNumAPC
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


'''Function version v10.0'''
def UnpackBits(in_intAr, Nbits):
    ''' convert (numpyarray of uint => array of Nbits bits) for many bits in parallel'''
    inSize_T = in_intAr.shape
    in_intAr_flat = in_intAr.flatten()
    out_NbitAr = np.zeros((len(in_intAr_flat), Nbits))
    for iBits in range(Nbits):
        out_NbitAr[:, iBits] = (in_intAr_flat >> iBits) & 1
    out_NbitAr = out_NbitAr.reshape(inSize_T + (Nbits,))
    return out_NbitAr


'''Function version v10.0'''
def GenerateLookupTableForAPC16(twoByte):
    twoByte = np.array(twoByte)
    # Represent the decimal value into 16bit binary value
    x = UnpackBits(twoByte, 16).astype('uint16')
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


'''Function version v10.0'''
def NOT(a):
    if(a == 0):
        return 1
    elif(a == 1):
        return 0


'''Function version v10.0'''
def GenerateLookupTableForAPC16Inversed(twoByte):
    twoByte = np.array(twoByte)
    # Represent the decimal value into 16bit binary value
    x = UnpackBits(twoByte, 16).astype('uint16')
    x = x[::-1]

    # Initialize the sum
    sum = 0

    # NAND, NOR gates
    a = NOT(x[0] | x[1])
    b = NOT(x[2] & x[3])
    c = NOT(x[4] | x[5])
    d = NOT(x[6] & x[7])
    e = NOT(x[8] | x[9])
    f = NOT(x[10] & x[11])
    z2 = (x[12] | x[13])
    t0 = NOT(x[14] & x[15])

    # Inversed Full Adder 1 (Carry:x1, Sum:x2)
    x1 = NOT((a & b) | (b & c) | (c & a))
    x2 = NOT((a ^ b) ^ c)

    # Inversed Full Adder 2 (Carry:y1, Sum:y2)
    y1 = NOT((d & e) | (e & f) | (f & d))
    y2 = NOT((d ^ e) ^ f)

    # Inversed Full Adder 3 (Carry:z1, Sum:t1)
    z1 = ((x2 & y2) | (y2 & z2) | (z2 & x2))
    t1 = NOT((x2 ^ y2) ^ z2)

    # Inversed Full Adder 4 (Carry:t3, Sum:t2)
    t3 = NOT((x1 & y1) | (y1 & z1) | (z1 & x1))
    t2 = NOT((x1 ^ y1) ^ z1)

    # Represent in the binary format
    # sum = 8 * t3 + 4 * t2 + 2 * t1 + 2 * t0
    # sum = 16 - sum

    t0 = NOT(t0)
    t1 = NOT(t1)
    t2 = NOT(t2)
    t3 = NOT(t3)
    sum = 8 * t3 + 4 * t2 + 2 * t1 + 2 * t0

    return sum


'''Function version v10.0'''
def GenerateLookupTableForAPC25Inversed(Bits):
    Bits = np.array(Bits)
    # Represent the decimal value into 16bit binary value
    x = UnpackBits(Bits, 25).astype('uint32')
    x = x[::-1]

    # Initialize the sum
    sum = 0

    # NAND, NOR gates
    a = NOT(x[0] | x[1])
    b = NOT(x[2] & x[3])
    c = NOT(x[4] | x[5])
    d = NOT(x[6] & x[7])
    e = NOT(x[8] | x[9])
    f = NOT(x[10] & x[11])
    g = NOT(x[12] | x[13])
    h = NOT(x[14] & x[15])
    i = NOT(x[16] & x[17])
    j = NOT(x[18] | x[19])
    k = NOT(x[20] & x[21])
    l = NOT(x[22] & x[23])
    t0 = NOT(x[24])

    # Inversed Full Adder 1 (Carry:m, Sum:n)
    m = NOT((a & b) | (b & c) | (c & a))
    n = NOT((a ^ b) ^ c)

    # Inversed Full Adder 2 (Carry:o, Sum:p)
    o = NOT((d & e) | (e & f) | (f & d))
    p = NOT((d ^ e) ^ f)

    # Inversed Full Adder 3 (Carry:q, Sum:r)
    q = NOT((g & h) | (h & i) | (i & g))
    r = NOT((g ^ h) ^ i)

    # Inversed Full Adder 4 (Carry:s, Sum:t)
    s = NOT((j & k) | (k & l) | (l & j))
    t = NOT((j ^ k) ^ l)

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
    sum = 16 * t4 + 8 * t3 + 4 * t2 + 2 * t1 + 1 * t0
    sum = 30 - sum

    # t0 = NOT(t0)
    # t1 = NOT(t1)
    # t2 = NOT(t2)
    # t3 = NOT(t3)
    # t4 = NOT(t4)
    # sum = 16 * t4 + 8 * t3 + 4 * t2 + 2 * t1 + 1 * t0

    return sum


# Generate the lookup table for 8bit, 16bit and 25bit APC
snLookupTableNumAPC8 = np.array([GenerateLookupTableForAPC8(Byte) for Byte in range(256)])
snLookupTableNumAPC16 = np.array([GenerateLookupTableForAPC16(twoByte) for twoByte in range(65536)])
snLookupTableNumAPC16Inversed = np.array([GenerateLookupTableForAPC16Inversed(twoByte) for twoByte in range(65536)])
snLookupTableNumAPC25 = np.array([GenerateLookupTableForAPC25Inversed(twoByte) for twoByte in range(33554432)])

with open('snLookupTableNumAPC.pkl', 'wb') as output:
    pkl_snLookupTableNumAPC8 = snLookupTableNumAPC8
    pickle.dump(pkl_snLookupTableNumAPC8, output, pickle.HIGHEST_PROTOCOL)

    pkl_snLookupTableNumAPC16 = snLookupTableNumAPC16
    pickle.dump(pkl_snLookupTableNumAPC16, output, pickle.HIGHEST_PROTOCOL)

    pkl_snLookupTableNumAPC16Inversed = snLookupTableNumAPC16Inversed
    pickle.dump(pkl_snLookupTableNumAPC16Inversed, output, pickle.HIGHEST_PROTOCOL)

    pkl_snLookupTableNumAPC25 = snLookupTableNumAPC25
    pickle.dump(pkl_snLookupTableNumAPC25, output, pickle.HIGHEST_PROTOCOL)