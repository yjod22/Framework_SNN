
from keras.callbacks import Callback
import global_variables
import numpy as np


class WeightScale(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global_variables.bnModel
        global_variables.cntEpochs += 1
        if(global_variables.bnModel.GetId() == 1):
            global_variables.bnModel.Save_weights('../results/#Epoch' + str(global_variables.cntEpochs)+'_weights_of_1st_model_network_test.h5')
        elif(global_variables.bnModel.GetId() == 2):
            global_variables.bnModel.Save_weights('../results/#Epoch' + str(global_variables.cntEpochs)+'_weights_of_2nd_model_network_test.h5')

    def on_batch_end(self, batch, logs=None):
        weights_conv1 = self.model.get_layer(index=1).get_weights()
        weights_conv2 = self.model.get_layer(index=5).get_weights()
        maximum = 1
        for w in weights_conv1:
            maximum = max(np.max(np.absolute(w)), maximum)
        maximumConv2 = 1
        for w in weights_conv2:
            maximumConv2 = max(np.max(np.absolute(w)), maximumConv2)
        self.model.get_layer(index=1).set_weights([w/maximum for w in weights_conv1])
        self.model.get_layer(index=5).set_weights([w/maximumConv2 for w in weights_conv2])