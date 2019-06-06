#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     large_nn_test.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   hybrid_cnn_passau.py, large_nn_test.py
# Version:     5.4 (SCR_V5.3-7)
# Author/Date: Junseok Oh / 2018-11-27
# Change:      Use Binary function for the 2nd dense layer through utils.py
# Cause:       Two convolutional layers and Two dense layers are applied
# Initiator:   Florian Neugebauer
################################################################################
# File:		   large_nn_test.py
# Version:     5.4 (SCR_V5.3-5)
# Author/Date: Junseok Oh / 2018-11-22
# Change:      Define scaling factors in the activation functions of Normal NN 
# Cause:       The scale of addition is needed to be taken into account
# Initiator:   Florian Neugebau
################################################################################
# File:		   large_nn_test.py
# Version:     5.4 (SCR_V5.3-4)
# Author/Date: Junseok Oh / 2018-11-20
# Change:      The activation functions for the conv layers are Tanh
#			   The activation function for the last dense layer is softmax
# Cause:       Define the correct activation function
# Initiator:   Florian Neugebauer
################################################################################
# File:		   large_nn_test.py
# Version:     5.4 (SCR_V5.3-2)
# Author/Date: Junseok Oh / 2018-11-15
# Change:      Add additional print statements around the softmax activation
# Cause:       Verification
# Initiator:   Florian Neugebauer
################################################################################
# File:		   hybrid_cnn_passau.py, large_nn_test.py, test.py
# Version:     5.4 (SCR_V5.3-1)
# Author/Date: Junseok Oh / 2018-11-14
# Change:      Use conventional binary function for the 2nd dense layer
# Cause:       The results of 1st dense layer isn't in the range between -1 and +1
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.3 (SCR_V5.2-1)
# Author/Date: Junseok Oh / 2018-11-09
# Change:      Check K.image_data_format() to determine shape of the image data
# Cause:       Linux & Windows Environment differences
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.2 (SCR_V5.1-2)
# Author/Date: Junseok Oh / 2018-10-28
# Change:      Make functions that extract weights and biases from the model
# Cause:       Multiple layers require a number of extraction of weights and biases
# Initiator:   Florian Neugebauer
################################################################################
# Version:     Initial version
# Author/Date: Junseok Oh / 2018-10-18
# Change:      Change parameter for light test
#              train:100, test:8, 1st Con2D:3, 2nd Conv2D:6, 1st Dense:10, length:1024
# Cause:       Hybrid Convolution Neural Network
# Initiator:   Florian Neugebauer
################################################################################
# Version:     Initial version
# Author/Date: Florian Neugebauer / 2018-10-18
# Change:      Initial version
# Cause:       Hybrid Convolution Neural Network
# Initiator:   Dr. Ilia Polian
###############################################################################
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import WeightScaling_large
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from snn.holayer import HOModel, HOMaxPoolingExact, HOMaxPoolingAprox, HOConvolution, HOConnected
from snn.utils import HOUtils

# misc functions:


def createSN(x, length):
    """create bipolar SN by comparing random vector elementwise to SN value x"""
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
    #return K.tanh(x*0.16)
    # 2 = 1 input layer x 1x1 filter + 1 bias
    return K.tanh(x/2)


def second_layer_activation(x):
    #return K.tanh(x*0.008)
    # 2 = 1 input layers x 1x1 filter + 1 bias
    return K.tanh(x/2)



def ActivationFuncRelu(x):
    '''input x is not a stochastic number'''
    if (x <= 0):
        return 0
    else:
        return x


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
model.add(Conv2D(1, kernel_size=(1, 1),
                 input_shape=input_shape))  # with MUX
model.add(Activation('first_layer_activation'))  # relu activation
model.add(Conv2D(10, kernel_size=(1, 1)))  # with MUX
model.add(Activation('second_layer_activation'))  # relu activation
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

#model.save_weights('test_result_v5.4_large.h5')
model.load_weights('test_result_v5.4_large.h5')
# model.load_weights('C:/Users/neugebfn/PycharmProjects/SCNN/weights/mnist_model_max_dropout.h5')

score = model.evaluate(x_test[:10000], y_test[:10000], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate(x_test[:100], y_test[:100], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

layer7model = Model(inputs=model.input, outputs=model.get_layer(index=7).output)
layer8model = Model(inputs=model.input, outputs=model.get_layer(index=8).output)
layer9model = Model(inputs=model.input, outputs=model.get_layer(index=9).output)
layer10model = Model(inputs=model.input, outputs=model.get_layer(index=10).output)

# SC Model from here:

# SN length
length = 1024*4

# Create the class of utility
ut = HOUtils()

# weights and biases of the convolutional layer
'''
TODO:
Need to extract weights and biases from the model here and generate SNs for them, e.g:

weights = model.get_layer(index=1).get_weights()[0]
biases = model.get_layer(index=1).get_weights()[1]
......

'''
bias_1_SNs = ut.GetConvolutionLayerBiasesSN(model, 1, length)
weight_1_SNs = ut.GetConvolutionLayerWeightsSN(model, 1, length)

bias_3_SNs = ut.GetConvolutionLayerBiasesSN(model, 3, length)
weight_3_SNs = ut.GetConvolutionLayerWeightsSN(model, 3, length)

dense_7_biases = ut.GetConnectedLayerBiases(model, 7)
dense_7_weight_SNs = ut.GetConnectedLayerWeightsSN(model, 7, length)

#Currently, it cannot perform the 2nd dense layer with the stochastic numbers due to the range of 1st dense layer results
dense_9_biases = ut.GetConnectedLayerBiases(model, 9)
dense_9_weights = ut.GetConnectedLayerWeights(model, 9)


SN_input_matrix = np.full((img_rows, img_cols, length), False)

correct_predictions = 0
test_index = 0
output_mse = 0

print('start stochastic NN')
# for each input in the test set
for r in range(10):
    x = x_test[test_index]
    print(test_index)
    # build input SN matrix
    for i in range(img_rows):
        for j in range(img_cols):
            if K.image_data_format() == 'channels_first':
                SN_input_matrix[i, j] = createSN(x[0, i, j], length)
            else:
                SN_input_matrix[i, j] = createSN(x[i, j, 0], length)

    print('inputs generated')

    '''
    TODO:
    HO MODEL COMES HERE
    '''


    hoModel = HOModel(SN_input_matrix)

    hoModel.SetNumOutputPlanes(1) # The number of slices:1
    hoModel.SetWeights(weight_1_SNs)
    hoModel.SetBias(bias_1_SNs)
    hoModel.Activation(HOConvolution(1, 1, length, activationFunc="Tanh"), stride=1) # Stride:1, filterSize:5x5, Activation function: Relu
    print('conv layer 1 done')

    hoModel.SetNumOutputPlanes(10) # The number of slices:10
    hoModel.SetWeights(weight_3_SNs)
    hoModel.SetBias(bias_3_SNs)
    hoModel.Activation(HOConvolution(1, 1, length, activationFunc="Tanh"), stride=1) # Stride:1, filterSize:5x5, Activation function: Relu
    print('conv layer 2 done')

    hoModel.Activation(HOMaxPoolingExact(2, 2, length), stride=2) # Stride:2, filterSize:2x2
    print('max pool 1 done')

    hoModel.SetNumOutputPlanes(1) # The number of slices:1
    hoModel.SetDenseWeights(dense_7_weight_SNs)
    hoModel.SetDenseBias(dense_7_biases)
    hoModel.Activation(HOConnected(length, stochToInt="APC", activationFunc="Relu"), num_classes=100)
    dense_output = hoModel.GetOutputMatrix()
    print('dense 1 layer done')
    print("Keras Prediction of dense 1 layer")
    print(layer7model.predict(np.asarray([x_test[test_index]])))
    print("SNN results of dense 1 layer")
    print(dense_output)

    dense_output = ut.BinaryConnectedLAyer(100, num_last_classes, dense_output, dense_9_weights, dense_9_biases)
    print('dense 2 layer done without the softmax activation function')
    print("Keras Prediction of dense 2 layer")	
    print(layer9model.predict(np.asarray([x_test[test_index]])))
    print("SNN results of dense 2 layer")
    print(dense_output)


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

correct_predictions = correct_predictions/10
print("correct classifications:", correct_predictions)
output_mse = output_mse/10
print("output_mse:", output_mse)