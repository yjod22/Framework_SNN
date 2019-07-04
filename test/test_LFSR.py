#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     test_LFSR.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   test_LFSR.py
# Version:     14.0
# Author/Date: Junseok Oh / 2019-07-01
# Change:      (SCR_V13.0-3): Implement Random Number Generator
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
### test different stanh circuits
import numpy as np
import plotly as py
import plotly.graph_objs as go
from snn.snn import HOSnn


# Initialize the graphs' data and parameters
SN_length = 2 ** 10
numSamples = 1000

result = np.zeros(numSamples)
reference = np.zeros(numSamples)

# get some random input values and sort them. These are going to be assigned to the x-axis
values = np.zeros(numSamples)
values = np.random.random(numSamples)*2*1 - 1*1
values = np.sort(values, 0)

# Refer to the class
hoSnn =HOSnn(kBits=10)

# produce the SNs for input values
SNs = np.full((numSamples, SN_length), False)

for i in range(values.shape[0]):
    # SNs[i] = hoSnn.CreateSNWithNumpy(values[i], SN_length)
    SNs[i] = hoSnn.CreateSN(values[i])

# Calculate the graphs' data that are going to be assigned to the y-axis
for i in range(values.shape[0]):
    result[i] = hoSnn.StochToInt(SNs[i])
    reference[i] = values[i]

# Assign the graphs' data to x-axis and y-axis
# CreateSN_y_x = go.Scatter(x=values, y=result, mode='markers', name='CreateSN_y_x')
CreateSN_y_x = go.Scatter(x=values, y=result, mode='markers', name='CreateSN_y_x')
y_x = go.Scatter(x=values, y=reference, name='y = x')

# Integrate the graphs' data
data = [CreateSN_y_x, y_x]

# create plots
py.offline.plot(data, filename='testplot_LFSR.html')
