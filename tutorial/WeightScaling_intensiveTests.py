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
# Version:     6.0 (SCR_V5.4-2)
# Author/Date: Junseok Oh / 2019-01-05
# Change:      This software is branched from v6.0-PreV07-WeightScaling.py
# Cause:       Intensive Stochastic Neural Network tests
# Initiator:   Florian Neugebauer
################################################################################

from keras.callbacks import Callback
import numpy as np


class WeightScale(Callback):

    def on_batch_end(self, batch, logs=None):
        weights_conv = self.model.get_layer(index=1).get_weights()
        #weights_dense = self.model.get_layer(index=6).get_weights()
        #weights_dense2 = self.model.get_layer(index=8).get_weights()
        maximum = 1
        for w in weights_conv:
            maximum = max(np.max(np.absolute(w)), maximum)
        #for w in weights_dense:
        #    maximum = max(np.max(np.absolute(w)), maximum)
        #for w in weights_dense2:
        #    maximum = max(np.max(np.absolute(w)), maximum)
        self.model.get_layer(index=1).set_weights([w/maximum for w in weights_conv])
        #self.model.get_layer(index=6).set_weights([w/maximum for w in weights_dense])
        #self.model.get_layer(index=8).set_weights([w/maximum for w in weights_dense2])

