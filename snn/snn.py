#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     snn.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   snn.py
# Version:     14.0
# Author/Date: Junseok Oh / 2019-07-01
# Change:      (SCR_V13.0-1): Place CreateSN on the higher class
#              (SCR_V13.0-2): Place StochToInt on the higher class
#              (SCR_V13.0-3): Implement Random Number Generator
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################

import numpy as np
import pickle
import copy

class HOSnn(object):
    def __init__(self, **kwargs):
        # Set kBits
        self.kBits = 10
        for key in kwargs:
            if (key == "kBits"):
                self.kBits = kwargs[key]

        # Set the length of stochastic number
        self.snLength = 2 ** self.kBits

        # Load LookUpTables for LFSRs
        with open('../listLFSRTable.pkl', 'rb') as input:
            listLFSRTable = pickle.load(input)

        # Select the specific LFSR LUT with given kBits
        self.listLFSR = listLFSRTable[self.kBits-1]
        del(listLFSRTable)

    def StochToInt(self, x):
        """convert bipolar stochastic number to integer"""
        return (sum(x) / len(x)) * 2.0 - 1.0

    def CreateSN(self, x):
        # Initialize the SN
        x_SN = np.full(self.snLength, False)

        # Get random numbers from the LFSR
        r = copy.deepcopy(self.listLFSR)
        r.append(r[0])

        # Rotate the order of LFSR's states
        n = np.random.randint(1, len(r))
        r = r[-n:] + r[:-n]

        # Convert Bipolar to Unipolar
        if(x > 1 or x < -1):
            print("The number is out of range (-1, +1)")
        b = (x + 1) / 2

        # Comparator (output is 1 if R < B)
        for i in range(self.snLength):
            if(r[i] < b):
                x_SN[i] = True

        return x_SN

    def _CreateSN(self, x):
        """create bipolar SN by comparing random vector elementwise to SN value x"""
        # rand = np.random.rand(length)*2.0 - 1.0
        #  x_SN = np.less(rand, x)
        large = np.random.rand(1)
        x_SN = np.full(self.snLength, False)
        if large:
            for i in range(int(np.ceil(((x + 1) / 2) * self.snLength))):
                try:
                    x_SN[i] = True
                except IndexError:
                    print("The number is out of range (-1, +1)")
                    print("x: "+ str(x))
        else:
            for i in range(int(np.floor(((x + 1) / 2) * self.snLength))):
                try:
                    x_SN[i] = True
                except IndexError:
                    print("The number is out of range (-1, +1)")
                    print("x: "+ str(x))
        np.random.shuffle(x_SN)
        return x_SN