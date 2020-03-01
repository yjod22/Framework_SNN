###############################################################################
#                                                                             #
#                            Copyright (c)                                    #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:	hoSnn.py
#  Description:	
#  Author/Date:	Junseok Oh / 2020-02-27
#  Initiator:	Florian Neugebauer
################################################################################
	
import numpy as np
import pickle
import copy


class HOSnn(object):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kBits: int
            the length of Stochastic Number (SN)
            e.g. kBits=10, the length of SN is 2^10
            e.g. kBits=11, the length of SN is 2^11
        """
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
        """
        Converting bipolar stochastic number to integer
        """
        return (sum(x) / len(x)) * 2.0 - 1.0

    def CreateSN(self, x):
        """
        (LFSR-based)
        Creating bipolar SN by comparing random vector elementwise to SN value x
        """
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
        """
        (ARCHIVED), It is replaced by the LFSR-based CreateSN
        Creating bipolar SN by comparing random vector elementwise to SN value x
        """
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