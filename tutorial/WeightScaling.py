#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     WeightScaling.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# Version:     5.4
# Author/Date: Junseok Oh / 2018-11-27
# Change:      Change parameter for small test
#              train:60000, test:800, 1st Con2D:9,
#              1st Dense:100, 2nd Dense:10, length:4096
# Cause:       Small test
# Initiator:   Florian Neugebauer
################################################################################

from keras.callbacks import Callback
import numpy as np


class WeightScale(Callback):

    def on_batch_end(self, batch, logs=None):
        weights_conv = self.model.get_layer(index=1).get_weights()
        weights_dense = self.model.get_layer(index=5).get_weights()
        weights_dense2 = self.model.get_layer(index=7).get_weights()
        maximum = 1
        for w in weights_conv:
            maximum = max(np.max(np.absolute(w)), maximum)
        for w in weights_dense:
            maximum = max(np.max(np.absolute(w)), maximum)
        for w in weights_dense2:
            maximum = max(np.max(np.absolute(w)), maximum)
        self.model.get_layer(index=1).set_weights([w/maximum for w in weights_conv])
        self.model.get_layer(index=5).set_weights([w/maximum for w in weights_dense])
        self.model.get_layer(index=7).set_weights([w/maximum for w in weights_dense2])
        #weights_dense = self.model.get_layer(index=5).get_weights()
        #for i in range(2704):
        #    for j in range(10):
        #        if np.absolute(weights_dense[0][i, j]) < 0.0004:
        #            weights_dense[0][i, j] = 0
        #self.model.get_layer(index=5).set_weights([w for w in weights_dense])
