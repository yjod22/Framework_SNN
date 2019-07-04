#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     generate_listLFSRTable.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   generate_listLFSRTable.py
# Version:     14.0
# Author/Date: Junseok Oh / 2019-07-03
# Change:      (SCR_V13.0-3): Implement Random Number Generator
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
import pickle

def GenerateLFSRTable(kBits, listPolynomials):
    # e.g.
    # 11 (in decimal)
    # 0b1011 (in binary format)
    # 0.6875 = 0.5 + 0.125 + 0.0625 (in decimal fraction)

    n = 2 ** kBits
    seed = 1
    listSeed = []
    for i in range(n - 1):
        seed = LFSR(seed, kBits, listPolynomials)
        listSeed.append(ConvertToDecimalFraction(seed, kBits))

    return listSeed


def LFSR(seed, kBits, listPolynomials):
    filter = (2 ** kBits) - 1

    # Given kBits and its polynomial, extract the bits which will be fed into XORs
    listXor = []
    for i in range(len(listPolynomials)):
        try:
            listXor.append((filter & (seed << (listPolynomials[i] - 1))) >> (kBits - 1))
        except ValueError:
            pass

    # Perform XOR operations consecutively with the extracted bits
    resultXor = 0
    for i in range(len(listXor)):
        resultXor = listXor[i] ^ resultXor

    # Shift right to one bit on the seed
    data = seed >> 1

    # Feed the result of consecutive XOR operations into the data
    resultXor = resultXor << (kBits - 1)
    result = data | resultXor

    return result


def ConvertToDecimalFraction(x, kBits):
    # Represent the number in the binary format
    listTemp = []
    for i in range(kBits):
        listTemp.append(x % 2)
        x = int(x / 2)
    listTemp = listTemp[::-1]

    # Convert to Decimal fraction
    listDecimalFraction = 0
    for i in range(len(listTemp)):
        listDecimalFraction += listTemp[i] / (2 ** (i + 1))

    return listDecimalFraction


# Generate the lookup table for LFSR with different kBits and its primitive polynomial
listLFSRTable = [0]
listLFSRTable.append(GenerateLFSRTable(2, [2, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(3, [3, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(4, [4, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(5, [5, 2, 0]))
listLFSRTable.append(GenerateLFSRTable(6, [6, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(7, [7, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(8, [8, 4, 3, 2, 0]))
listLFSRTable.append(GenerateLFSRTable(9, [9, 4, 0]))
listLFSRTable.append(GenerateLFSRTable(10, [10, 3, 0]))
listLFSRTable.append(GenerateLFSRTable(11, [11, 2, 0]))
listLFSRTable.append(GenerateLFSRTable(12, [12, 6, 4, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(13, [13, 4, 3, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(14, [14, 8, 6, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(15, [15, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(16, [16, 12, 3, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(17, [17, 3, 0]))
listLFSRTable.append(GenerateLFSRTable(18, [18, 7, 0]))
listLFSRTable.append(GenerateLFSRTable(19, [19, 6, 2, 1, 0]))
listLFSRTable.append(GenerateLFSRTable(20, [20, 3, 0]))

with open('listLFSRTable.pkl', 'wb') as output:
    pkl_listLFSRTable = listLFSRTable
    pickle.dump(pkl_listLFSRTable, output, pickle.HIGHEST_PROTOCOL)
