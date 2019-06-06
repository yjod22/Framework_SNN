#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     WeightScaling_intensiveTests.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   WeightScaling_intensiveTests.py
# Version:     8.0 
# Author/Date: Junseok Oh / 2019-05-23
# Change:      Create on_epoch_end
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   WeightScaling_intensiveTests.py
# Version:     6.0 (SCR_V5.4-2)
# Author/Date: Junseok Oh / 2019-01-05
# Change:      This software is branched from v6.0-PreV07-WeightScaling.py
# Cause:       Intensive Stochastic Neural Network tests
# Initiator:   Florian Neugebauer
################################################################################
import keras
from keras.utils.generic_utils import get_custom_objects
from keras.layers import MaxPooling2D, Activation, Flatten, Dense
from keras import backend as K
import global_variables

from keras.callbacks import Callback
import numpy as np


class WeightScale(Callback):

    def first_layer_activation_2(self, x):
        # return K.tanh(x)
        print("Relu in callback")
        return K.relu(x) / 2
        # return K.tanh(x/2.5)

    def first_layer_activation_10(self, x):
        # return K.tanh(x)
        print("Relu in callback")
        return K.relu(x) / 10
        # return K.tanh(x/2.5)

    def on_epoch_end(self, epoch, logs=None):
        global_variables.bnModel
        global_variables.cntEpochs += 1
        if(global_variables.bnModel.GetId() == 1):
            global_variables.bnModel.Save_weights('results/#Epoch' + str(global_variables.cntEpochs)+' weights of 1st model_tests57.h5')
        elif(global_variables.bnModel.GetId() == 2):
            global_variables.bnModel.Save_weights('results/#Epoch' + str(global_variables.cntEpochs)+' weights of 2nd model_tests57.h5')

    def on_batch_end(self, batch, logs=None):
        weights_conv = self.model.get_layer(index=1).get_weights()
        weights_conv2 = self.model.get_layer(index=3).get_weights()
        weights_dense = self.model.get_layer(index=7).get_weights()
        maximum = 1
        for w in weights_conv:
            maximum = max(np.max(np.absolute(w)), maximum)
        maximumConv2 = 1
        for w in weights_conv2:
            maximumConv2 = max(np.max(np.absolute(w)), maximumConv2)
        maximumDense = 1
        for w in weights_dense:
            maximumDense = max(np.max(np.absolute(w)), maximumDense)
        self.model.get_layer(index=1).set_weights([w/maximum for w in weights_conv])
        self.model.get_layer(index=3).set_weights([w/maximumConv2 for w in weights_conv2])
        self.model.get_layer(index=7).set_weights([w/maximumDense for w in weights_dense])