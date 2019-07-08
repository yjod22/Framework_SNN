#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     generate_snLookupTableNumRelu.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   generate_snLookupTableNumRelu.py
# Version:     15.0
# Author/Date: Junseok Oh / 2019-07-04
# Change:      (SCR_V14.0-2): Generate the LUT for Relu outside
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   generate_snLookupTableNumAPC.py
# Version:     12.0
# Author/Date: Junseok Oh / 2019-06-25
# Change:      (SCR_V11.0-1): Fix bug in NOR, NAND gates
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################

import numpy as np
import pickle


def GenerateLookupTableForRelu(byte):
    # Represent the decimal value into 8bit binary value
    x = np.unpackbits(np.array([byte], dtype='uint8'))

    # Initialize the number of one
    numOne = 0

    # Count up the number of one if the bit is equal to 1
    for j, bit in enumerate(x):
        if (bit == 1):
            numOne = numOne + 1

    return numOne

# Generate the lookup table for Relu activation function
# 8bit [11111111] is equal to 255 in decimal
snLookupTableNum = np.array([GenerateLookupTableForRelu(byte) for byte in range(256)])

with open('snLookupTableNumRelu.pkl', 'wb') as output:
    pkl_snLookupTableNum = snLookupTableNum
    pickle.dump(pkl_snLookupTableNum, output, pickle.HIGHEST_PROTOCOL)