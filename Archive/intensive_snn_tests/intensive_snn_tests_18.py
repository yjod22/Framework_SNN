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
# Version:     6.4
# Author/Date: Junseok Oh / 2019-03-24
# Change:      snLength: 4096
#              Conv(1, 4x4), activation function(tanh(0.7x)), Conv(1, 4x4), activation function(tanh(0.7x)),
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
    #return K.relu(x/10)

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
model.add(Conv2D(1, kernel_size=(4, 4),
                 input_shape=input_shape, use_bias=False))  # with APC
model.add(Activation('first_layer_activation'))  # tanh(x/2) activation
model.add(Conv2D(1, kernel_size=(4, 4), use_bias=False))  # with APC
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
#model.save_weights('v6.4_test_result_IntensiveTests_18.h5')
model.load_weights('v6.4_test_result_IntensiveTests_18.h5')
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
#layer8model = Model(inputs=model.input, outputs=model.get_layer(index=8).output)
layer9model = Model(inputs=model.input, outputs=model.get_layer(index=9).output)
#layer10model = Model(inputs=model.input, outputs=model.get_layer(index=10).output)

# Hybrid NN with stochastic convolutional layer and binary dense layer

# SN length
length = 1024*4
#length = 1024*4

ut = HOUtils()

# weights and biases of the convolutional layer
#bias_1_SNs = ut.GetConvolutionLayerBiasesSN(model, 1, length)
weight_1_SNs = ut.GetConvolutionLayerWeightsSN(model, 1, length)

#bias_3_SNs = ut.GetConvolutionLayerBiasesSN(model, 3, length)
weight_3_SNs = ut.GetConvolutionLayerWeightsSN(model, 3, length)

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
    hoModel.SetNumOutputPlanes(1) # The number of slices:1
    hoModel.SetWeights(weight_1_SNs)
    hoModel.SetZeroBias(1)
    hoConvLayer = HOConvolution(4, 4, length, baseMode="APC", activationFunc="BTanh", use_bias="False")
    hoModel.Activation(hoConvLayer, stride=1)
    del(hoConvLayer)

    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    # Extract the intermediate results from the model
    conv_output = hoModel.GetOutputMatrix()

    # Convert Stochastic number to Binary number
    conv_out_test = np.zeros((1, 25, 25))
    for i in range(1):
        for j in range(25):
            for k in range(25):
                conv_out_test[i, j, k] = stochtoint(conv_output[i, j, k])
    del(conv_output)

    # Predict the intermediate results from the Binary Neural Network
    BNN_prediction = layer2model.predict(np.asarray([x_test[test_index]]))

    # Write the array to disk
    txtTitle = 'v6.4_intensive_snn_tests18_conv_SNN1_' + str(test_index+1) + '.txt'
    with open(txtTitle, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(conv_out_test.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in conv_out_test:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.3f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    del (conv_out_test)

    txtTitle = 'v6.4_intensive_snn_tests18_conv_BNN1_' + str(test_index+1) + '.txt'
    with open(txtTitle, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(BNN_prediction[0].shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in BNN_prediction[0]:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.3f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

    del(BNN_prediction)
    ###################################################################################################################
    print('conv layer 1 done')

    # convolutional layer 2
    hoModel.SetNumOutputPlanes(1) # The number of slices:1
    hoModel.SetWeights(weight_3_SNs)
    hoModel.SetZeroBias(1)
    hoConvLayer = HOConvolution(4, 4, length, baseMode="APC", activationFunc="BTanh", use_bias="False")
    hoModel.Activation(hoConvLayer, stride=1)
    del(hoConvLayer)

    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    # Extract the intermediate results from the model
    conv_output = hoModel.GetOutputMatrix()

    # Convert Stochastic number to Binary number
    conv_out_test = np.zeros((1, 22, 22))
    for i in range(1):
        for j in range(22):
            for k in range(22):
                conv_out_test[i, j, k] = stochtoint(conv_output[i, j, k])
    del(conv_output)

    # Predict the intermediate results from the Binary Neural Network
    BNN_prediction = layer4model.predict(np.asarray([x_test[test_index]]))

    # Write the array to disk
    txtTitle = 'v6.4_intensive_snn_tests18_conv_SNN2_' + str(test_index+1) + '.txt'
    with open(txtTitle, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(conv_out_test.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in conv_out_test:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.3f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    del (conv_out_test)

    txtTitle = 'v6.4_intensive_snn_tests18_conv_BNN2_' + str(test_index+1) + '.txt'
    with open(txtTitle, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(BNN_prediction[0].shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in BNN_prediction[0]:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.3f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

    del(BNN_prediction)
    ###################################################################################################################
    print('conv layer 2 done')

    # max pooling layer
    hoMaxLayer = HOMaxPoolingExact(2, 2, length)
    hoModel.Activation(hoMaxLayer, stride=2) # Stride:2, filterSize:2x2
    del(hoMaxLayer)

    ################### For debugging purpose, save the intermidiate results into the externel files ###################
    # Extract the intermediate results from the model
    maxpool_output = hoModel.GetOutputMatrix()

    # Convert Stochastic number to Binary number
    max_pool_out_test = np.zeros((1, 11, 11))
    for i in range(1):
        for j in range(11):
            for k in range(11):
                max_pool_out_test[i, j, k] = stochtoint(maxpool_output[i, j, k])
    del(maxpool_output)

    # Predict the intermediate results from the Binary Neural Network
    BNN_prediction = layer5model.predict(np.asarray([x_test[test_index]]))

    # Write the array to disk
    txtTitle = 'v6.4_intensive_snn_tests18_maxpool_SNN_' + str(test_index+1) + '.txt'
    with open(txtTitle, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(max_pool_out_test.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in max_pool_out_test:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.3f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    del(max_pool_out_test)

    txtTitle = 'v6.4_intensive_snn_tests18_maxpool_BNN_' + str(test_index+1) + '.txt'
    with open(txtTitle, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(BNN_prediction[0].shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in BNN_prediction[0]:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.3f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    del(BNN_prediction)
    ###################################################################################################################
    print('max pool 1 done')

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
    BNN_prediction = layer7model.predict(np.asarray([x_test[test_index]]))
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