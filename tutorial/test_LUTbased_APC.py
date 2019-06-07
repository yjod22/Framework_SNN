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
