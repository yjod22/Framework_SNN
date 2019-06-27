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
import keras
from keras.utils.generic_utils import get_custom_objects
from keras.layers import MaxPooling2D, Activation, Flatten, Dense
from keras import backend as K

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
        global cntEpochs
        global bnModel
        cntEpochs += 1
        bnModel.Save_weights('#Epoch' + str(cntEpochs)+' weights of 1st model_tests38.h5')

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

        #print("on_batch_end")
        #get_custom_objects().update({'first_layer_activation_10': Activation(self.first_layer_activation_10)})
        #self.model.pop()
        #self.model.pop()
        #self.model.pop()
        #self.model.pop()
        #self.model.pop()
        #self.model.add(Activation('first_layer_activation_10'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Flatten())
        #self.model.add(Dense(10)) # At the moment, the number of class is 10
        #self.model.add(Activation('softmax'))
        #self.model.get_layer(index=1).activation = intensive_snn_tests_1.first_layer_activation_1
        #self.model.get_layer(index=1).kernel_size = (2, 2)

        #self.model.compile(loss=keras.losses.mse,
        #              optimizer=keras.optimizers.Adadelta(),
        #              metrics=['accuracy'])
        #self.model.get_layer(index=4).activation = 'softmax'