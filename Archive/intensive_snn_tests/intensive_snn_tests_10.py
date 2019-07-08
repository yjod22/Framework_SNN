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
# Version:     6.1
# Author/Date: Junseok Oh / 2019-03-24
# Change:      snLength: 10240
#              Conv(16, 3x3), activation function(relu(x/10)), maxPooling(2x2), Dense, activation function(softmax)
#              Stochastic Conv(Mux+Relu), Stochastic Dense(Normal mode)
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
import WeightScaling_intensiveTests
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from snn.holayer import HOModel, HOMaxPoolingExact, HOMaxPoolingAprox, HOConvolution, HOConnected
from snn.utils import HOUtils

# misc functions
def first_layer_activation(x):
    #return K.tanh(x)
    return K.relu(x)/10
    #return K.tanh(x/2.5)


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

get_custom_objects().update({'first_layer_activation': Activation(first_layer_activation)})

np.set_printoptions(threshold=np.inf)

batch_size = 128
num_classes = 10
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

# Binary NN for reference
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 input_shape=input_shape))
model.add(Activation('first_layer_activation'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=0,
#          callbacks=[WeightScaling_intensiveTests.WeightScale()],
#          validation_data=(x_test, y_test))
#model.save_weights('v6.1_test_result_IntensiveTests_10.h5')
model.load_weights('v6.1_test_result_IntensiveTests_10.h5')
score = model.evaluate(x_test[:500], y_test[:500], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = model.evaluate(x_test[:107], y_test[:107], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

layer1model = Model(inputs=model.input, outputs=model.get_layer(index=1).output)
layer2model = Model(inputs=model.input, outputs=model.get_layer(index=2).output)
layer3model = Model(inputs=model.input, outputs=model.get_layer(index=3).output)
layer4model = Model(inputs=model.input, outputs=model.get_layer(index=4).output)
layer5model = Model(inputs=model.input, outputs=model.get_layer(index=5).output)
layer6model = Model(inputs=model.input, outputs=model.get_layer(index=6).output)
#layer7model = Model(inputs=model.input, outputs=model.get_layer(index=7).output)
#layer8model = Model(inputs=model.input, outputs=model.get_layer(index=8).output)
#layer9model = Model(inputs=model.input, outputs=model.get_layer(index=9).output)

# Hybrid NN with stochastic convolutional layer and binary dense layer

# SN length
length = 1024*10
#length = 1024*4

ut = HOUtils()

# weights and biases of the convolutional layer
bias_SNs = ut.GetConvolutionLayerBiasesSN(model, 1, length)
weight_SNs = ut.GetConvolutionLayerWeightsSN(model, 1, length)

# weights and biases of dense layer
dense_biases = ut.GetConnectedLayerBiases(model, 5)
dense_weight_SNs = ut.GetConnectedLayerWeightsSN(model, 5, length)

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

    # convolutional layer
    hoModel.SetNumOutputPlanes(16) # The number of slices:16
    hoModel.SetWeights(weight_SNs)
    hoModel.SetBias(bias_SNs)
    hoConvLayer = HOConvolution(3, 3, length, baseMode="Mux", activationFunc="Relu")
    hoModel.Activation(hoConvLayer, stride=1)
    del(hoConvLayer)

    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    # Extract the intermediate results from the model
    conv_output = hoModel.GetOutputMatrix()

    # Convert Stochastic number to Binary number
    conv_out_test = np.zeros((16, 26, 26))
    for i in range(16):
        for j in range(26):
            for k in range(26):
                conv_out_test[i, j, k] = stochtoint(conv_output[i, j, k])
    del(conv_output)

    # Predict the intermediate results from the Binary Neural Network
    BNN_prediction = layer2model.predict(np.asarray([x_test[test_index]]))

    # Write the array to disk
    txtTitle = 'v6.1_intensive_snn_tests10_conv_SNN_' + str(test_index+1) + '.txt'
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

    txtTitle = 'v6.1_intensive_snn_tests10_conv_BNN_' + str(test_index+1) + '.txt'
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
    print('conv layer done')

    # max pooling layer
    hoMaxLayer = HOMaxPoolingExact(2, 2, length)
    hoModel.Activation(hoMaxLayer, stride=2) # Stride:2, filterSize:2x2
    del(hoMaxLayer)

    ################### For debugging purpose, save the intermidiate results into the externel files ###################
    # Extract the intermediate results from the model
    maxpool_output = hoModel.GetOutputMatrix()

    # Convert Stochastic number to Binary number
    max_pool_out_test = np.zeros((16, 13, 13))
    for i in range(16):
        for j in range(13):
            for k in range(13):
                max_pool_out_test[i, j, k] = stochtoint(maxpool_output[i, j, k])
    del(maxpool_output)

    # Predict the intermediate results from the Binary Neural Network
    BNN_prediction = layer3model.predict(np.asarray([x_test[test_index]]))

    # Write the array to disk
    txtTitle = 'v6.1_intensive_snn_tests10_maxpool_SNN_' + str(test_index+1) + '.txt'
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

    txtTitle = 'v6.1_intensive_snn_tests10_maxpool_BNN_' + str(test_index+1) + '.txt'
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
    print('max pool layer done')

    # First dense layer
    hoModel.SetDenseWeights(dense_weight_SNs)
    hoModel.SetDenseBias(dense_biases)
    hoDenseLayer = HOConnected(length, stochToInt="Normal", activationFunc="None")
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
    ################################################################################################################
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

    if(np.argmax(hybrid_output) == np.argmax(y_test[test_index])):
        correct_predictions = correct_predictions + 1
    test_index = test_index + 1

    current_accuracy = correct_predictions/test_index

    print(current_accuracy)

    del(dense_output)
    del(hoModel)


correct_predictions = correct_predictions/10
print("correct classifications:", correct_predictions)
output_mse = output_mse/10
print("output_mse:", output_mse)