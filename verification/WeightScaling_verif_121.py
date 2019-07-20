#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     WeightScaling_verif_121.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   WeightScaling_verif_121.py
# Version:     16.0
# Author/Date: Junseok Oh / 2019-07-18
# Change:      (SCR_V15.0-2): Update more test cases
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
import global_variables
from keras.callbacks import Callback
import numpy as np

class WeightScale(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global_variables.bnModel
        global_variables.cntEpochs += 1
        if(global_variables.bnModel.GetId() == 1):
            global_variables.bnModel.Save_weights('../results/#Epoch' + str(global_variables.cntEpochs)+'_weights_of_1st_model_verif_121.h5')
        elif(global_variables.bnModel.GetId() == 2):
            global_variables.bnModel.Save_weights('../results/#Epoch' + str(global_variables.cntEpochs)+'_weights_of_2nd_model_verif_121.h5')

    def on_batch_end(self, batch, logs=None):
        weights_conv = self.model.get_layer(index=1).get_weights()
        weights_conv2 = self.model.get_layer(index=4).get_weights()
        weights_dense = self.model.get_layer(index=8).get_weights()
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
        self.model.get_layer(index=4).set_weights([w/maximumConv2 for w in weights_conv2])
        self.model.get_layer(index=8).set_weights([w/maximumDense for w in weights_dense])