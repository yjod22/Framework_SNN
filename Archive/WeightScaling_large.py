#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     WeightScaling_large.py
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
# Change:      Change parameter for large test
#              train:60000, test:800, 1st Con2D:1, 2nd Conv2D:10,
#              1st Dense:100, 2nd Dense:10, length:4096
# Cause:       Large test
# Initiator:   Florian Neugebauer
################################################################################

from keras.callbacks import Callback
import numpy as np


class WeightScale(Callback):

    def on_batch_end(self, batch, logs=None):
        weights_conv1 = self.model.get_layer(index=1).get_weights()
        weights_conv2 = self.model.get_layer(index=3).get_weights()
        weights_dense1 = self.model.get_layer(index=6).get_weights()
        weights_dense2 = self.model.get_layer(index=8).get_weights()
        maximum = 1
        for w in weights_conv1:
            maximum = max(np.max(np.absolute(w)), maximum)
        for w in weights_conv2:
            maximum = max(np.max(np.absolute(w)), maximum)
        for w in weights_dense1:
            maximum = max(np.max(np.absolute(w)), maximum)
        for w in weights_dense2:
            maximum = max(np.max(np.absolute(w)), maximum)
        self.model.get_layer(index=1).set_weights([w/maximum for w in weights_conv1])
        self.model.get_layer(index=3).set_weights([w / maximum for w in weights_conv2])
        self.model.get_layer(index=6).set_weights([w / maximum for w in weights_dense1])
        self.model.get_layer(index=8).set_weights([w / maximum for w in weights_dense2])
