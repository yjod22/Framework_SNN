#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     WeightScaling_verif_91.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   WeightScaling_verif_91.py
# Version:     15.0
# Author/Date: Junseok Oh / 2019-07-04
# Change:      (SCR_V14.0-5): Verify the all functionality
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
            global_variables.bnModel.Save_weights('../results/#Epoch' + str(global_variables.cntEpochs)+'_weights_of_1st_model_verif_91.h5')
        elif(global_variables.bnModel.GetId() == 2):
            global_variables.bnModel.Save_weights('../results/#Epoch' + str(global_variables.cntEpochs)+'_weights_of_2nd_model_verif_91.h5')

    def on_batch_end(self, batch, logs=None):
        weights_dense = self.model.get_layer(index=2).get_weights()
        maximumDense = 1
        for w in weights_dense:
            maximumDense = max(np.max(np.absolute(w)), maximumDense)
        self.model.get_layer(index=2).set_weights([w/maximumDense for w in weights_dense])