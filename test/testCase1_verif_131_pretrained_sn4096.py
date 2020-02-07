#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     testCase1_verif_131_pretrained_sn4096.py
#
###############################################################################
#  Description:
#  
#  (For a detailed description look at the object description in the UML model)
#  
###############################################################################
# History
################################################################################
# File:        testCaseX_verif_131, WeightScaling_testCaseX_verif_131.py
# Version:     19.2
# Author/Date: Junseok Oh / 2020-02-04
# Change:      (SCR_V19.1-3): Extend the simulation of verif_131
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:        verif_131, hoModel, holayer, hoUtils.py
# Version:     18.3
# Author/Date: Junseok Oh / 2019-11-26
# Change:      (SCR_V18.2-1): Use stochastic numbers for the dense layer's biases
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:        verif_131.py
# Version:     18.0
# Author/Date: Junseok Oh / 2019-07-21
# Change:      (SCR_V17.0-1): Update more test cases
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers
import WeightScaling_testCase5_verif_131
import numpy as np
from sklearn.utils import shuffle
from snn.hoModel import HOModel
from snn.hoLayer import HOMaxPoolingExact, HOMaxPoolingAprox, HOConvolution, HOConnected
from snn.hoUtils import HOUtils
from snn.bnLayer import BNModel
import global_variables

# misc functions
def first_layer_activation(x):
    return K.tanh(x)

#get_custom_objects().update({'first_layer_activation': Activation(first_layer_activation)})

np.set_printoptions(threshold=np.inf)

batch_size = 128
num_classes = 10
epochs = 20
cntEpochs = 0

# Define the global variables
global_variables.DefineGlobalVariables()

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train[:60000]
x_test = x_test[:800]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = y_train[:60000]
y_test = y_test[:800]
print(y_train.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 1st Model
global_variables.bnModel = BNModel(9)
global_variables.bnModel.SetId(1) # Set as the 1st model
global_variables.bnModel[0] = Conv2D(8, kernel_size=(4, 4),
                                     input_shape=input_shape,
                                     use_bias=False,
                                     kernel_regularizer=regularizers.l1(0))
global_variables.bnModel[1] = Activation(first_layer_activation)
global_variables.bnModel[2] = MaxPooling2D(pool_size=(2, 2))
global_variables.bnModel[3] = Conv2D(8, kernel_size=(5, 5),
                                     input_shape=input_shape,
                                     use_bias=False,
                                     kernel_regularizer=regularizers.l1(0))
global_variables.bnModel[4] = Activation(first_layer_activation)
global_variables.bnModel[5] = MaxPooling2D(pool_size=(2, 2))
global_variables.bnModel[6] = Flatten()
global_variables.bnModel[7] = Dense(num_classes)
global_variables.bnModel[8] = Activation('softmax')
global_variables.bnModel.LoadLayers()
global_variables.bnModel.Compile(loss=keras.losses.mse,
                                 optimizer=keras.optimizers.Adadelta(),
                                 metrics=['accuracy'])
#global_variables.bnModel.Fit(x_train, y_train,
#                             batch_size=batch_size,
#                             epochs=epochs,
#                             verbose=0,
#                             callbacks=[WeightScaling_testCase5_verif_131.WeightScale()],
#                             validation_data=(x_test, y_test))
global_variables.bnModel.Load_weights('../results/#Epoch20_weights_of_1st_model_verif_131_NotOptNotL1.h5')
global_variables.bnModel.Evaluate(x_test[:800], y_test[:800], verbose=0, indexModel=1)

# Get the layer models from bnModel
layer1model = global_variables.bnModel[0]
layer2model = global_variables.bnModel[1]
layer3model = global_variables.bnModel[2]
layer4model = global_variables.bnModel[3]
layer5model = global_variables.bnModel[4]
layer6model = global_variables.bnModel[5]
layer7model = global_variables.bnModel[6]
layer8model = global_variables.bnModel[7]
layer9model = global_variables.bnModel[8]

# Hybrid NN with stochastic convolutional layer and binary dense layer

# SN length
kBits = 12
length = 2 ** kBits

ut = HOUtils(kBits=kBits)
model = global_variables.bnModel.GetModel()
global_variables.bnModel = 0

# weights and biases of the convolutional layer
weight_1_SNs, bias_1_SNs, listIndex1 = ut.GetConvolutionLayerWeightsBiasesSN(model, 1, Adaptive="True")
weight_2_SNs, bias_2_SNs, listIndex2 = ut.GetConvolutionLayerWeightsBiasesSN(model, 4, Adaptive="True")

#dense_biases = ut.GetConnectedLayerBiases(model, 8)
#dense_weight_SNs = ut.GetConnectedLayerWeightsSN(model, 8)
dense_weight_SNs, dense_biases_SNs, listIndexDense = ut.GetConnectedLayerWeightsBiasesSN(model, 8, Adaptive="True")

# shuffle the test data samples
x_test, y_test = shuffle(x_test, y_test)

correct_predictions = 0
test_index = 0
output_mse = 0
iter_validation = 800

print('start stochastic NN')
# for each input in the test set
for r in range(iter_validation):
    x = x_test[test_index]
    print(test_index)

    # build input SN matrix
    SN_input_matrix = np.full((img_rows, img_cols, length), False)
    for i in range(img_rows):
        for j in range(img_cols):
            SN_input_matrix[i, j] = ut.CreateSN(x[0, i, j])
    del(x)
    print('inputs generated')

    # Generate the HOModel
    hoModel = HOModel(SN_input_matrix, kBits=kBits)
    del(SN_input_matrix)

    # convolutional layer 1
    hoModel.SetNumOutputPlanes(8) # The number of slices:8
    hoModel.SetWeights(weight_1_SNs)
    hoModel.SetZeroBias(8)
    hoModel.SetListIndex(listIndex1)
    #hoModel.SetBias(bias_1_SNs)
    hoConvLayer = HOConvolution(4, 4, kBits=kBits, baseMode="Mux", activationFunc="STanh", use_bias="False")

    hoModel.Activation(hoConvLayer, stride=1)
    del(hoConvLayer)
    print('conv layer 1 done')

    if(test_index % 100 == 0):
        ut.SaveInTxtFormat('../results/v19.2_testCase1_verif_131_conv1', test_index,
                           hoModel.GetOutputMatrix(), 8, 25, 25,
                           layer2model, x_test)
        print(str(test_index + 1) + ' conv 1 layer results saved in txt format')

    # max pooling layer
    hoMaxLayer = HOMaxPoolingExact(2, 2, kBits=kBits)
    hoModel.Activation(hoMaxLayer, stride=2)  # Stride:2, filterSize:2x2
    del (hoMaxLayer)
    print('max pool 1 done')

    # convolutional layer 2
    hoModel.SetNumOutputPlanes(8)  # The number of slices:8
    hoModel.SetWeights(weight_2_SNs)
    hoModel.SetZeroBias(8)
    hoModel.SetListIndex(listIndex2)
    #hoModel.SetBias(bias_2_SNs)
    hoConvLayer = HOConvolution(5, 5, kBits=kBits, baseMode="Mux", activationFunc="STanh", use_bias="False",
                                scale=1, constantH=0.8)
    hoModel.Activation(hoConvLayer, stride=1)
    del (hoConvLayer)
    print("conv layer 2 done")

    if (test_index % 100 == 0):
        ut.SaveInTxtFormat('../results/v19.2_testCase1_verif_131_conv2', test_index,
                           hoModel.GetOutputMatrix(), 8, 8, 8,
                           layer5model, x_test)
        print(str(test_index + 1) + ' conv layer 2 results saved in txt format')

    # max pooling layer 2
    hoMaxLayer = HOMaxPoolingExact(2, 2, kBits=kBits)
    hoModel.Activation(hoMaxLayer, stride=2) # Stride:2, filterSize:2x2
    del(hoMaxLayer)
    print('max pool 2 done')

    # First dense layer
    hoModel.SetNumOutputPlanes(1) # The number of slices:1
    hoModel.SetDenseWeights(dense_weight_SNs)
    hoModel.SetDenseBias(dense_biases_SNs)
    hoModel.SetListIndexDense(listIndexDense)
    hoDenseLayer = HOConnected(kBits=kBits, stochToInt="APC", activationFunc="None", use_bias="True")
    hoModel.Activation(hoDenseLayer, num_classes=num_classes)
    del(hoDenseLayer)
    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    dense_output = hoModel.GetOutputMatrix()
    print("dense 1 output from Binary NN without softmax")
    BNN_prediction = layer8model.predict(np.asarray([x_test[test_index]]))
    print(BNN_prediction)
    del(BNN_prediction)
    print("dense 1 output from Stochastic NN without softmax")
    print(dense_output)
    ###################################################################################################################
    print('dense 1 layer done')

    out_error = 0
    out = layer8model.predict(np.asarray([x_test[test_index]]))
    for i in range(10):
        out_error = out_error + (dense_output[0, i] - out[0, i])**2

    print("Current output_mse:", out_error)
    output_mse = output_mse + out_error

    # softmax activation
    dense_out_exp = [np.exp(i) for i in dense_output]
    exp_sum_out = np.sum(dense_out_exp)
    hybrid_output = [i/exp_sum_out for i in dense_out_exp]
    print('dense done with the softmax activation function')
    print("Labeled output of the dense layer")
    print(np.argmax(y_test[test_index]))
    print("SNN results of dense layer")
    print(np.argmax(hybrid_output))

    if(np.argmax(hybrid_output) == np.argmax(y_test[test_index])):
        correct_predictions = correct_predictions + 1
    test_index = test_index + 1

    current_accuracy = correct_predictions/test_index

    print('current_accuracy')
    print(current_accuracy)

    del(dense_output)
    del(hoModel)

correct_predictions = correct_predictions/iter_validation
print("correct classifications:", correct_predictions)
output_mse = output_mse/iter_validation
print("Average of output_mse:", output_mse)
