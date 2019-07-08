#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     verif_11.py
#
###############################################################################
#  Description:
#  
#  (For a detailed description look at the object description in the UML model)
#  
###############################################################################
# History
################################################################################
# File:		   verif_11.py
# Version:     15.0
# Author/Date: Junseok Oh / 2019-07-01
# Change:      (SCR_V14.0-5): Verify the all functionality
# Cause:       -
# Initiator:   Florian Neugebauer
###############################################################################
import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from verification import WeightScaling_verif_11
import numpy as np
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
epochs = 10
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
global_variables.bnModel = BNModel(6)
global_variables.bnModel.SetId(1) # Set as the 1st model
global_variables.bnModel[0] = Conv2D(4, kernel_size=(4, 4),
                                     input_shape=input_shape,
                                     use_bias=True)
global_variables.bnModel[1] = Activation(first_layer_activation)
global_variables.bnModel[2] = MaxPooling2D(pool_size=(2, 2))
global_variables.bnModel[3] = Flatten()
global_variables.bnModel[4] = Dense(num_classes)
global_variables.bnModel[5] = Activation('softmax')
global_variables.bnModel.LoadLayers()
global_variables.bnModel.Compile(loss=keras.losses.mse,
                                 optimizer=keras.optimizers.Adadelta(),
                                 metrics=['accuracy'])
global_variables.bnModel.Fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=0,
                             callbacks=[WeightScaling_verif_11.WeightScale()],
                             validation_data=(x_test, y_test))
global_variables.bnModel.Load_weights('../results/#Epoch10_weights_of_1st_model_verif_11.h5')
global_variables.bnModel.Evaluate(x_test[:500], y_test[:500], verbose=0, indexModel=1)
global_variables.bnModel.Evaluate(x_test[:1000], y_test[:1000], verbose=0, indexModel=1)

# Get the layer models from bnModel
layer1model = global_variables.bnModel[0]
layer2model = global_variables.bnModel[1]
layer3model = global_variables.bnModel[2]
layer4model = global_variables.bnModel[3]
layer5model = global_variables.bnModel[4]
layer6model = global_variables.bnModel[5]

# Hybrid NN with stochastic convolutional layer and binary dense layer

# SN length
kBits = 10
length = 2 ** kBits

ut = HOUtils(kBits=kBits)
model = global_variables.bnModel.GetModel()
global_variables.bnModel = 0

# weights and biases of the convolutional layer
weight_1_SNs, bias_1_SNs, listIndex1 = ut.GetConvolutionLayerWeightsBiasesSN(model, 1, Adaptive="False")

dense_5_biases = ut.GetConnectedLayerBiases(model, 5)   
dense_5_weight_SNs = ut.GetConnectedLayerWeightsSN(model, 5)

output = np.zeros((1, 10))
correct_predictions = 0
test_index = 0
output_mse = 0

print('start stochastic NN')
# for each input in the test set
for r in range(10):
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
    hoModel.SetNumOutputPlanes(4) # The number of slices:4
    hoModel.SetWeights(weight_1_SNs)
    #hoModel.SetZeroBias(2)
    hoModel.SetListIndex(listIndex1)
    hoModel.SetBias(bias_1_SNs)
    hoConvLayer = HOConvolution(4, 4, kBits=kBits, baseMode="APC", use_bias="True",
                                activationFunc="BTanh", scale=1, constantH=0.7)
    hoModel.Activation(hoConvLayer, stride=1)
    del(hoConvLayer)
    print('conv layer 1 done')

    if(test_index % 10 == 0):
        ut.SaveInTxtFormat('../results/v15.0_verif_11_conv1', test_index,
                           hoModel.GetOutputMatrix(), 4, 25, 25,
                           layer2model, x_test)
        print(str(test_index + 1) + ' conv 1 layer results saved in txt format')

    # max pooling layer
    hoMaxLayer = HOMaxPoolingExact(2, 2, kBits=kBits)
    hoModel.Activation(hoMaxLayer, stride=2) # Stride:2, filterSize:2x2
    del(hoMaxLayer)
    print('max pool 1 done')
    if(test_index % 10 == 0):
        ut.SaveInTxtFormat('../results/v15.0_verif_11_maxpool', test_index,
                           hoModel.GetOutputMatrix(), 4, 12, 12,
                           layer3model, x_test)
        print(str(test_index+1)+' maxpool layer results saved in txt format')

    # First dense layer
    hoModel.SetNumOutputPlanes(1) # The number of slices:1
    hoModel.SetDenseWeights(dense_5_weight_SNs)
    hoModel.SetDenseBias(dense_5_biases)
    hoDenseLayer = HOConnected(kBits=kBits, stochToInt="Normal", activationFunc="None", use_bias="True")
    hoModel.Activation(hoDenseLayer, num_classes=num_classes)
    del(hoDenseLayer)
    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    dense_output = hoModel.GetOutputMatrix()
    print("dense 1 output from Binary NN")
    BNN_prediction = layer5model.predict(np.asarray([x_test[test_index]]))
    print(BNN_prediction)
    del(BNN_prediction)
    print("dense 1 output from Stochastic NN")
    print(dense_output)
    ###################################################################################################################
    print('dense 1 layer done')

    out_error = 0
    out = layer5model.predict(np.asarray([x_test[test_index]]))
    for i in range(10):
        out_error = out_error + (dense_output[0, i] - out[0, i])**2

    print("out_error:", out_error)
    output_mse = output_mse + out_error

    # softmax activation
    dense_out_exp = [np.exp(i) for i in dense_output]
    exp_sum_out = np.sum(dense_out_exp)
    hybrid_output = [i/exp_sum_out for i in dense_out_exp]
    print('dense done with the softmax activation function')
    print("Keras Prediction of max argument of dense layer")
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

correct_predictions = correct_predictions/10
print("correct classifications:", correct_predictions)
output_mse = output_mse/10
print("output_mse:", output_mse)