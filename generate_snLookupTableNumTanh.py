#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     generate_snLookupTableNumTanh.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:        generate_snLookupTableNumTanh.py
# Version:     19.1
# Author/Date: Junseok Oh / 2019-12-05
# Change:      (SCR_V19.0-1): Change max. scale factor to 200
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   generate_snLookupTableNumTanh.py
# Version:     15.0
# Author/Date: Junseok Oh / 2019-07-04
# Change:      (SCR_V14.0-3): Generate the LUT for Tanh outside
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
import copy


def GenerateLookupTableForSTanh(byte, start_state, PAR_numState):
    # Set the number of states
    numState = PAR_numState

    # Represent the decimal value into 8bit binary value
    x = np.unpackbits(np.array([byte], dtype='uint8'))

    # Set state to start_state
    state = start_state

    for j, bit in enumerate(x):
        # Determine the output depending on the current state
        x[j] = state > (numState / 2 - 1)

        # input is True -> increase state
        if state < (numState - 1) and bit:
            state += 1
        # input is False -> decrease state
        if state > 0 and not bit:
            state -= 1

    return (np.packbits(x)[0], state)

# Generate the lookup table for STanh activation function
# the number of scale factors is 20. i.e. tanh(1*x), tanh(2*x), ... , tanh(200*x)
# the number of states of STanh's state machine is determined by (num+1)*2
# 8bit [11111111] is equal to 255 in decimal
numScaleFactor = 200
snLookupTableOut = [[]for num in range(numScaleFactor)]
snLookupTableState = [[]for num in range(numScaleFactor)]
for num in range(numScaleFactor):
    snLookupTableElementsTemp = np.array(
        [[GenerateLookupTableForSTanh(byte, state, (num+1)*2) for byte in range(256)] for state in range((num+1)*2)])
    snLookupTableOut[num] = copy.deepcopy(snLookupTableElementsTemp[:, :, 0])
    snLookupTableState[num] = copy.deepcopy(snLookupTableElementsTemp[:, :, 1])

with open('snLookupTableNumTanh.pkl', 'wb') as output:
    pkl_snLookupTableOut = snLookupTableOut
    pickle.dump(pkl_snLookupTableOut, output, pickle.HIGHEST_PROTOCOL)

    pkl_snLookupTableState = snLookupTableState
    pickle.dump(pkl_snLookupTableState, output, pickle.HIGHEST_PROTOCOL)