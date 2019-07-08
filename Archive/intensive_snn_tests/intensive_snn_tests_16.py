#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     intensive_snn_tests.py
#
###############################################################################
#  Description:
#  
#  (For a detailed description look at the object description in the UML model)
#  
###############################################################################
# History
################################################################################
# File:		   intensive_snn_tests.py
# Version:     8.0
# Author/Date: Junseok Oh / 2019-05-23
# Change:      snLength: 4096
#              Conv(2, 4x4), activation function(tanh(0.7x)), Conv(3, 4x4), activation function(tanh(0.7x)),
#              maxPooling(2x2), Dense(100), activation function(relu), Dense(10), activation function(softmax)
#              Stochastic Conv(APC+BTanh), Stochastic Conv(APC+BTanh), Stochastic Dense(APC mode+Relu), BinaryConnectedLAyer
#			   (SCR_V6.4-1): NN Optimization-JSO (Make use of listIndex not to consider zero weights in addition)
#			   (SCR_V6.4-4): Create SaveInTxtFormat function
# Cause:       -
# Initiator:   Junseok Oh
################################################################################
# File:		   intensive_snn_tests.py
# Version:     6.4
# Author/Date: Junseok Oh / 2019-03-24
# Change:      snLength: 4096
#              Conv(2, 4x4), activation function(tanh(0.7x)), Conv(3, 4x4), activation function(tanh(0.7x)),
#              maxPooling(2x2), Dense(100), activation function(relu), Dense(10), activation function(softmax)
#              Stochastic Conv(APC+BTanh), Stochastic Conv(APC+BTanh), Stochastic Dense(APC mode+Relu), BinaryConnectedLAyer
# Cause:       Need short description for this file
# Initiator:   Junseok Oh
################################################################################
# File:		   intensive_snn_tests.py
# Version:     6.1 (SCR_V6.0-5)
# Author/Date: Junseok Oh / 2019-01-31
# Change:      Save the intermediate results in the txt format
#			   Refer to the following website
#			   https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file/3685295
# Cause:       Need to extract the intermediate results
# Initiator:   Florian Neugebauer
################################################################################
# File:		   intensive_snn_tests.py
# Version:     6.1 (SCR_V6.0-4)
# Author/Date: Junseok Oh / 2019-01-31
# Change:      Delete the object when it is not needed anymore
# Cause:       Need to handle the memory leakage issue during runtime
# Initiator:   Florian Neugebauer
################################################################################
# File:		   intensive_snn_tests.py
# Version:     6.0 (SCR_V5.4-2)
# Author/Date: Junseok Oh / 2019-01-05
# Change:      This software is branched from v6.0-PreV07-hybrid_cnn_passau.py
# Cause:       Intensive Stochastic Neural Network tests
# Initiator:   Florian Neugebauer
###############################################################################
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import WeightScaling_large
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from snn.holayer import HOModel, HOMaxPoolingExact, HOMaxPoolingAprox, HOConvolution, HOConnected
from snn.utils import HOUtils

# misc functions


def createSN(x, length):
    """create bipolar SN by comparing random vector elementwise to SN value x"""
    # rand = np.random.rand(length)*2.0 - 1.0
    # x_SN = np.less(rand, x)
    large = np.random.rand(1)
    x_SN = np.full(length, False)
    if large:
        for i in range(int(np.ceil(((x+1)/2)*length))):
            x_SN[i] = True
    else:
        for i in range(int(np.floor(((x+1)/2)*length))):
            x_SN[i] = True
    np.random.shuffle(x_SN)
    return x_SN

def stochtoint(x):
    """convert bipolar stochastic number to integer"""
    return (sum(x)/len(x))*2.0 - 1.0

def first_layer_activation(x):
    return K.tanh(x*0.7)
    # 2 = 1 input layer x 1x1 filter + 1 bias
    #return K.tanh(x/2)
    #return K.relu(x/2)



def second_layer_activation(x):
    return K.tanh(x*0.7)
    # 2 = 1 input layers x 1x1 filter + 1 bias
    #return K.tanh(x/2)
    #return K.relu(x/2)

get_custom_objects().update({'first_layer_activation': Activation(first_layer_activation)})
get_custom_objects().update({'second_layer_activation': Activation(second_layer_activation)})

np.set_printoptions(threshold=np.inf)

batch_size = 128
num_last_classes = 10
epochs = 2

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
y_train = keras.utils.to_categorical(y_train, num_last_classes)
y_test = keras.utils.to_categorical(y_test, num_last_classes)
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

# Binary NN for reference
model = Sequential()
model.add(Conv2D(2, kernel_size=(4, 4),
                 input_shape=input_shape, use_bias=False))  # with APC
model.add(Activation('first_layer_activation'))  # tanh(x/2) activation
model.add(Conv2D(3, kernel_size=(4, 4), use_bias=False))  # with APC
model.add(Activation('second_layer_activation'))  # tanh(x/2) activation
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100))  # with APC
model.add(Activation('relu'))
model.add(Dense(num_last_classes))  # with APC
model.add(Activation('softmax'))

model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=0,
#          callbacks=[WeightScaling_large.WeightScale()],
#          validation_data=(x_test, y_test))
#model.save_weights('v6.4_test_result_IntensiveTests_16.h5')
model.load_weights('v6.4_test_result_IntensiveTests_16.h5')
score = model.evaluate(x_test[:500], y_test[:500], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = model.evaluate(x_test[:107], y_test[:107], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#layer1model = Model(inputs=model.input, outputs=model.get_layer(index=1).output)
layer2model = Model(inputs=model.input, outputs=model.get_layer(index=2).output)
#layer3model = Model(inputs=model.input, outputs=model.get_layer(index=3).output)
layer4model = Model(inputs=model.input, outputs=model.get_layer(index=4).output)
layer5model = Model(inputs=model.input, outputs=model.get_layer(index=5).output)
#layer6model = Model(inputs=model.input, outputs=model.get_layer(index=6).output)
layer7model = Model(inputs=model.input, outputs=model.get_layer(index=7).output)
layer8model = Model(inputs=model.input, outputs=model.get_layer(index=8).output)
layer9model = Model(inputs=model.input, outputs=model.get_layer(index=9).output)
#layer10model = Model(inputs=model.input, outputs=model.get_layer(index=10).output)

# Hybrid NN with stochastic convolutional layer and binary dense layer

# SN length
length = 1024*4
#length = 1024*4

ut = HOUtils()

# weights and biases of the convolutional layer
#bias_1_SNs = ut.GetConvolutionLayerBiasesSN(model, 1, length)
#weight_1_SNs = ut.GetConvolutionLayerWeightsSN(model, 1, length)
weight_1_SNs, bias_1_SNs, listIndex1 = ut.GetConvolutionLayerWeightsBiasesSN(model, 1, length, Adaptive="False")


#bias_3_SNs = ut.GetConvolutionLayerBiasesSN(model, 3, length)
#weight_3_SNs = ut.GetConvolutionLayerWeightsSN(model, 3, length)
weight_3_SNs, bias_3_SNs, listIndex3 = ut.GetConvolutionLayerWeightsBiasesSN(model, 3, length, Adaptive="False")


dense_7_biases = ut.GetConnectedLayerBiases(model, 7)
dense_7_weight_SNs = ut.GetConnectedLayerWeightsSN(model, 7, length)

#Currently, it cannot perform the 2nd dense layer with the stochastic numbers due to the range of 1st dense layer results
dense_9_biases = ut.GetConnectedLayerBiases(model, 9)
dense_9_weights = ut.GetConnectedLayerWeights(model, 9)
#SN_input_matrix = np.full((img_rows, img_cols, length), False)

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
            SN_input_matrix[i, j] = createSN(x[0, i, j], length)
    del(x)
    print('inputs generated')

    # Generate the HOModel
    hoModel = HOModel(SN_input_matrix)
    del(SN_input_matrix)

    # convolutional layer 1
    hoModel.SetNumOutputPlanes(2) # The number of slices:2
    hoModel.SetWeights(weight_1_SNs)
    hoModel.SetZeroBias(2)
    hoModel.SetListIndex(listIndex1)
    hoConvLayer = HOConvolution(4, 4, length, baseMode="APC", activationFunc="BTanh", use_bias="False")
    hoModel.Activation(hoConvLayer, stride=1)
    del(hoConvLayer)
    print('conv layer 1 done')

    ut.SaveInTxtFormat('v8.0_intensive_snn_tests16_conv1', test_index,
                       hoModel.GetOutputMatrix(), 2, 25, 25,
                       layer2model, x_test)
    print(str(test_index + 1) + ' conv 1 layer results saved in txt format')

    # convolutional layer 2
    hoModel.SetNumOutputPlanes(3) # The number of slices:3
    hoModel.SetWeights(weight_3_SNs)
    hoModel.SetZeroBias(3)
    hoModel.SetListIndex(listIndex3)
    hoConvLayer = HOConvolution(4, 4, length, baseMode="APC", activationFunc="BTanh", use_bias="False")
    hoModel.Activation(hoConvLayer, stride=1)
    del(hoConvLayer)
    print('conv layer 2 done')

    ut.SaveInTxtFormat('v8.0_intensive_snn_tests16_conv2', test_index,
                       hoModel.GetOutputMatrix(), 3, 22, 22,
                       layer4model, x_test)
    print(str(test_index + 1) + ' conv 2 layer results saved in txt format')

    # max pooling layer
    hoMaxLayer = HOMaxPoolingExact(2, 2, length)
    hoModel.Activation(hoMaxLayer, stride=2) # Stride:2, filterSize:2x2
    del(hoMaxLayer)
    print('max pool 1 done')

    ut.SaveInTxtFormat('v8.0_intensive_snn_tests16_maxpool', test_index,
                       hoModel.GetOutputMatrix(), 3, 11, 11,
                       layer5model, x_test)
    print(str(test_index+1)+' maxpool layer results saved in txt format')

    # First dense layer
    hoModel.SetNumOutputPlanes(1) # The number of slices:1
    hoModel.SetDenseWeights(dense_7_weight_SNs)
    hoModel.SetDenseBias(dense_7_biases)
    hoDenseLayer = HOConnected(length, stochToInt="APC", activationFunc="Relu", use_bias="True")
    hoModel.Activation(hoDenseLayer, num_classes=100)
    del(hoDenseLayer)
    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    dense_output = hoModel.GetOutputMatrix()
    print("dense 1 output from Binary NN")
    BNN_prediction = layer8model.predict(np.asarray([x_test[test_index]]))
    print(BNN_prediction)
    del(BNN_prediction)
    print("dense 1 output from Stochastic NN")
    print(dense_output)
    ###################################################################################################################
    print('dense 1 layer done')

    # Second dense layer
    dense_output = hoModel.GetOutputMatrix()
    dense_output = ut.BinaryConnectedLAyer(100, num_last_classes, dense_output, dense_9_weights, dense_9_biases)

    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    print("dense 2 output from Binary NN")
    BNN_prediction = layer9model.predict(np.asarray([x_test[test_index]]))
    print(BNN_prediction)
    del(BNN_prediction)
    print("dense 2 output from Stochastic NN")
    print(dense_output)
    ###################################################################################################################
    print('dense 2 layer done')



    out_error = 0
    out = layer9model.predict(np.asarray([x_test[test_index]]))
    for i in range(10):
        out_error = out_error + (dense_output[0, i] - out[0, i])**2

    print("out_error:", out_error)
    output_mse = output_mse + out_error

    # softmax activation
    dense_out_exp = [np.exp(i) for i in dense_output]
    exp_sum_out = np.sum(dense_out_exp)
    hybrid_output = [i/exp_sum_out for i in dense_out_exp]
    print('dense 2 done with the softmax activation function')
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