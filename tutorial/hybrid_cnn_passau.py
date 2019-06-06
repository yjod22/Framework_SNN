#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     hybrid_cnn_passau.py
#
###############################################################################
#  Description:
#  
#  (For a detailed description look at the object description in the UML model)
#  
###############################################################################
# History
################################################################################
# File:		   hybrid_cnn_passau.py, //large_nn_test.py
# Version:     6.0 (SCR_V5.4-1)
# Author/Date: Junseok Oh / 2019-01-01
# Change:      Change activation function from Relu to STanh
# Cause:       Test stochastic hyperbolic tangent
# Initiator:   Florian Neugebauer
################################################################################
# File:		   hybrid_cnn_passau.py, large_nn_test.py
# Version:     5.4 (SCR_V5.3-7)
# Author/Date: Junseok Oh / 2018-11-27
# Change:      Use Binary function for the 2nd dense layer through utils.py
# Cause:       Two dense layers are applied
# Initiator:   Florian Neugebauer
################################################################################
# File:		   hybrid_cnn_passau.py, large_nn_test.py, test.py
# Version:     5.4 (SCR_V5.3-1)
# Author/Date: Junseok Oh / 2018-11-14
# Change:      Use conventional binary function for the 2nd dense layer
# Cause:       The results of 1st dense layer isn't in the range between -1 and +1
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.2 (SCR_V5.1-2)
# Author/Date: Junseok Oh / 2018-10-28
# Change:      Make functions that extract weights and biases from the model
# Cause:       Multiple layers require a number of extraction of weights and biases
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.1 (SCR_V5.0-1)
# Author/Date: Junseok Oh / 2018-10-02
# Change:	   Assign the layer ID (Convolution, MaxPooling, and FullyConnected)
#			   Define the numInputPlanes and numOutputPlanes
#			   Forward the whole planes as inputs for the Convolution layer
# Cause:       The number of slices has to be variable
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.0 (SCR_V4.0-1)
# Author/Date: Junseok Oh / 2018-09-10
# Change:	   Create the new HOConnected class
# Cause:       Implementation of fully connected layer with APC-16bit
# Initiator:   Florian Neugebauer
################################################################################
# Version:     4.0 (SCR_V3.0-1)
# Author/Date: Junseok Oh / 2018-08-22
# Change:	   Create the new ActivationFuncTanh
# Cause:       Implementation of stanh
# Initiator:   Florian Neugebauer
################################################################################
# Version:     3.0 (SCR_V2.1-1)
# Author/Date: Junseok Oh / 2018-08-14
# Change:	   Change the structure of classes
#              Create the new HOConvolution
# Cause:       Implementation of multiple layers
#              Implementation of convolution
# Initiator:   Florian Neugebauer
################################################################################
# Version:     2.0 (SCR_V1.2-1)
# Author/Date: Junseok Oh / 2018-07-17
# Change:	   Use the library of holayer version 2.0
# Cause:       Implementing Object-Oriented Exact Stochastic MaxPooling
# Initiator:   Florian Neugebauer
################################################################################ 
# Version:     1.1
# Author/Date: Junseok Oh / 2018-07-05
# Change:      Change the range of c-bits 
# Cause:       Bug fix in aprox_smax function 
# Initiator:   Junseok Oh
################################################################################
# Version:     1.0
# Author/Date: Florian Neugebauer / 2018-05-30
# Change:      Initial version
# Cause:       Hybrid Convolution Neural Network
# Initiator:   Dr. Ilia Polian
###############################################################################
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import WeightScaling
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from snn.holayer import HOModel, HOMaxPoolingExact, HOMaxPoolingAprox, HOConvolution, HOConnected
from snn.utils import HOUtils

# misc functions
def stanh(x):
    """activation function for stochastic NN. FSM with 8 states"""
    # starting state
    state = 3
    out = np.full(len(x), True, dtype=bool)
    for j in range(len(x)):
        # input is True -> increase state
        if x[j] & (state < 7):
            if state > 3:
                out[j] = 1
            else:
                out[j] = 0
            state = state + 1
        elif x[j] & (state == 7):
            out[j] = 1
        # input is False -> decrease state
        elif (np.logical_not(x[j])) & (state > 0):
            if state > 3:
                out[j] = 1
            else:
                out[j] = 0
            state = state - 1
        elif (np.logical_not(x[j])) & (state == 0):
            out[j] = 0
    return out


def srelu(x):
    if sum(x) > len(x)/2:
        return x
    else:
        return createSN(0, len(x))


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


def neuron(weights, inputs, bias):
    """stochastic neuron in convolutional layer. Gets SN weight matrix, SN input matrix and bias SN"""
    length = len(bias)
    result = np.full(length, False)
    # vector for products of input*weight + bias
    products = np.full((10, length), False)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            products[i*3 + j] = np.logical_not(np.logical_xor(weights[i, j], inputs[i, j]))
    products[9] = bias

    for i in range(length):
        r = np.floor(np.random.rand(1)[0]*10)
        result[i] = products[int(r), i]

    # apply stochastic activation function stanh
    #result = stanh(result)
    result = srelu(result)
    return result


def stochtoint(x):
    """convert bipolar stochastic number to integer"""
    return (sum(x)/len(x))*2.0 - 1.0


def first_layer_activation(x):
    return K.tanh(x)
    #return K.relu(x)/10
    #return K.tanh(x/2.5)


#def smax_pool(x):
#    values = np.zeros((2, 2))
#    for i in range(2):
#        for j in range(2):
#            values[i, j] = np.sum(x[i, j])
#    n, m = np.unravel_index(values.argmax(), values.shape)
#    return x[n, m]

def aprox_smax(x):
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    SN1 = x[0, 0]
    SN2 = x[0, 1]
    SN3 = x[1, 0]
    SN4 = x[1, 1]

    SN_length = len(SN1)

    SN_out = np.full((1, SN_length), False)

    step = 128

    for l in range(int(len(SN1)/step) - 1):
        counter1 = sum(SN1[(step * l):((l + 1) * step )])
        counter2 = sum(SN2[(step * l):((l + 1) * step )])
        counter3 = sum(SN3[(step * l):((l + 1) * step )])
        counter4 = sum(SN4[(step * l):((l + 1) * step )])

        max_count = max([counter1, counter2, counter3, counter4])

        if counter1 == max_count:
            SN_out[0, (step * (l + 1)):(((l + 1) + 1) * step )] = SN1[(step * (l + 1)):(((l + 1) + 1) * step )]
        if counter2 == max_count:
            SN_out[0, (step * (l + 1)):(((l + 1) + 1) * step )] = SN2[(step * (l + 1)):(((l + 1) + 1) * step )]
        if counter3 == max_count:
            SN_out[0, (step * (l + 1)):(((l + 1) + 1) * step )] = SN3[(step * (l + 1)):(((l + 1) + 1) * step )]
        if counter4 == max_count:
            SN_out[0, (step * (l + 1)):(((l + 1) + 1) * step )] = SN4[(step * (l + 1)):(((l + 1) + 1) * step )]

    initial = np.random.rand(1)*4
    initial = int(np.ceil(initial[0]))

    if initial == 1:
        SN_out[0, 0:(step )] = SN1[0:(step )]
    if initial == 2:
        SN_out[0, 0:(step )] = SN2[0:(step )]
    if initial == 3:
        SN_out[0, 0:(step )] = SN3[0:(step )]
    if initial == 4:
        SN_out[0, 0:(step )] = SN4[0:(step )]

    return SN_out


def smax_pool(x):
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    SN1 = x[0, 0]
    SN2 = x[0, 1]
    SN3 = x[1, 0]
    SN4 = x[1, 1]

    SN_length = len(SN1)

    SN_out = np.full((1, SN_length), False)

    for l in range(len(SN1)):
        out1 = SN1[l] & (counter1 == 0)
        out2 = SN2[l] & (counter2 == 0)
        out3 = SN3[l] & (counter3 == 0)
        out4 = SN4[l] & (counter4 == 0)

        SN_out[0, l] = out1 | out2 | out3 | out4

        Inc1 = np.logical_not(SN1[l]) & SN_out[0, l]
        Dec1 = SN1[l] & (counter1 != 0) & np.logical_not(out2 | out3 | out4)
        Inc2 = np.logical_not(SN2[l]) & SN_out[0, l]
        Dec2 = SN2[l] & (counter2 != 0) & np.logical_not(out1 | out3 | out4)
        Inc3 = np.logical_not(SN3[l]) & SN_out[0, l]
        Dec3 = SN3[l] & (counter3 != 0) & np.logical_not(out1 | out2 | out4)
        Inc4 = np.logical_not(SN4[l]) & SN_out[0, l]
        Dec4 = SN4[l] & (counter4 != 0) & np.logical_not(out1 | out2 | out3)

        if (Inc1):
            counter1 = counter1 + 1
        elif (Dec1):
            counter1 = counter1 - 1

        if (Inc2):
            counter2 = counter2 + 1
        elif (Dec2):
            counter2 = counter2 - 1

        if (Inc3):
            counter3 = counter3 + 1
        elif (Dec3):
            counter3 = counter3 - 1

        if (Inc4):
            counter4 = counter4 + 1
        elif (Dec4):
            counter4 = counter4 - 1

    return SN_out


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
model.add(Conv2D(9, kernel_size=(4, 4),
                 input_shape=input_shape))
model.add(Activation('first_layer_activation'))
#model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          callbacks=[WeightScaling.WeightScale()],
          validation_data=(x_test, y_test))
#model.save_weights('test_result_v5.4_small.h5')
#model.load_weights('test_result_v5.4_small.h5')
#model.load_weights('C:/Users/neugebfn/PycharmProjects/SCNN/weights/mnist_model_max_passau.h5')
score = model.evaluate(x_test[:500], y_test[:500], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = model.evaluate(x_test[:107], y_test[:107], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

layer4model = Model(inputs=model.input, outputs=model.get_layer(index=4).output)
l4out = layer4model.predict(np.asarray([x_test[0]]))
layer5model = Model(inputs=model.input, outputs=model.get_layer(index=5).output)
layer6model = Model(inputs=model.input, outputs=model.get_layer(index=6).output)
layer7model = Model(inputs=model.input, outputs=model.get_layer(index=7).output)
layer8model = Model(inputs=model.input, outputs=model.get_layer(index=8).output)
#flattenmodel = Model(inputs=model.input, outputs=model.get_layer(index=3).output)
#flout = flattenmodel.predict(np.asarray([x_test[0]]))

#print(l2out.shape)
#print('l2out:', l2out)


# Hybrid NN with stochastic convolutional layer and binary dense layer

# SN length
#length = 1024
length = 1024*4

ut = HOUtils()

# weights and biases of the convolutional layer
bias_SNs = ut.GetConvolutionLayerBiasesSN(model, 1, length)
weight_SNs = ut.GetConvolutionLayerWeightsSN(model, 1, length)

# weights and biases of dense layer
dense_biases = ut.GetConnectedLayerBiases(model, 6)
dense_weight_SNs = ut.GetConnectedLayerWeightsSN(model, 6, length)

dense_8_biases = ut.GetConnectedLayerBiases(model, 8)
dense_8_weights = ut.GetConnectedLayerWeights(model, 8)



SN_input_matrix = np.full((img_rows, img_cols, length), False)

conv_layer_output = np.full((16, 26, 26, length), False)
#conv_layer_output = np.zeros((16, 26, 26))
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
    for i in range(img_rows):
        for j in range(img_cols):
            SN_input_matrix[i, j] = createSN(x[0, i, j], length)

    print('inputs generated')
    # calculate output of convolutional layer
    #for i in range(16):
    #    #print('neuron:', i)
    #    for j in range(26):
    #        for k in range(26):
    #            conv_layer_output[i, j, k] = neuron(weight_SNs[i], SN_input_matrix[j:(j+3), k:(k+3)], bias_SNs[i])

    #conv_out_test = np.zeros((16, 26, 26))
    #for i in range(16):
    #   for j in range(26):
    #       for k in range(26):
    #           conv_out_test[i, j, k] = stochtoint(conv_layer_output[i, j, k])
    #print(conv_out_test)

    hoModel = HOModel(SN_input_matrix)
    hoModel.SetNumOutputPlanes(9) # The number of slices:9
    hoModel.SetWeights(weight_SNs)
    hoModel.SetBias(bias_SNs)
    hoModel.Activation(HOConvolution(4, 4, length, baseMode="APC", activationFunc="STanh"), stride=1)
    print('conv layer done')

    # max pooling layer
    #max_pool_out = np.zeros((16, 13, 13))
    #max_pool_out_SN = np.full((16, 13, 13, length), False)
    #for i in range(16):
    #    for j in range(13):
    #        for k in range(13):
    #            max_pool_out_SN[i, j, k] = aprox_smax(conv_layer_output[i, (2*j):(2*j+2), (2*k):(2*k+2)])[0]
                #smax_pool
    #            h1 = HOMaxPooling(conv_layer_output[i, (2*j):(2*j+2), (2*k):(2*k+2)], 2, 2, length)
    #            max_pool_out_SN[i, j, k] = h1.EncapsulatedFuncs()[0]

    #holayer1 = HOLayer(conv_layer_output, 2, 2)

    hoModel.Activation(HOMaxPoolingExact(2, 2, length), stride=2) # Stride:2, filterSize:2x2
    print('max pool done')

    #max_out_test = np.zeros((16, 13, 13))
    #for i in range(16):
    #    for j in range(13):
    #        for k in range(13):
    #            max_out_test[i, j, k] = stochtoint(max_pool_out_SN[i, j, k])
    #print(max_out_test)

    '''
    # calculate the dense layer output
    #dense_input = max_pool_out.reshape((1, 2704))
    dense_input_SN = max_pool_out_SN.reshape((1, 2704, length))
    #print(dense_input[0, 0:625])
    #dense_output = np.dot(dense_input, dense_weights) + dense_biases

    dense_output_SN = np.full((10, 2704, length), False)
    for i in range(2704):
        for j in range(10):
            dense_output_SN[j, i] = np.logical_not(np.logical_xor(dense_input_SN[0, i], dense_weight_SNs[i, j]))

    dense_output = np.zeros((1, 10))
    for i in range(10):
        for j in range(2704):
            dense_output[0, i] = dense_output[0, i] + stochtoint(dense_output_SN[i, j])

    dense_output = dense_output + dense_biases
    '''
    hoModel.SetDenseWeights(dense_weight_SNs)
    hoModel.SetDenseBias(dense_biases)
    hoModel.Activation(HOConnected(length, stochToInt="APC", activationFunc="Relu"), num_classes=100)
    dense_output = hoModel.GetOutputMatrix()
    print('dense 1 layer done')

    print(layer6model.predict(np.asarray([x_test[test_index]])))
    print("dense 1 output")
    print(dense_output)

    dense_output = ut.BinaryConnectedLAyer(100, num_classes, dense_output, dense_8_weights, dense_8_biases)
    print('dense 2 layer done')

    print(layer8model.predict(np.asarray([x_test[test_index]])))
    print("dense 2 output")
    print(dense_output)

    out_error = 0
    out = layer8model.predict(np.asarray([x_test[test_index]]))
    for i in range(10):
        out_error = out_error + (dense_output[0, i] - out[0, i])**2

    print("out_error:", out_error)

    output_mse = output_mse + out_error

    # softmax activation
    dense_out_exp = [np.exp(i) for i in dense_output]
    exp_sum_out = np.sum(dense_out_exp)
    hybrid_output = [i/exp_sum_out for i in dense_out_exp]

    #print(model.predict(np.asarray([x_test[test_index]])))
    #print(hybrid_output)

    if(np.argmax(hybrid_output) == np.argmax(y_test[test_index])):
        correct_predictions = correct_predictions + 1
    test_index = test_index + 1

    current_accuracy = correct_predictions/test_index

    print(current_accuracy)

correct_predictions = correct_predictions/10
print("correct classifications:", correct_predictions)
output_mse = output_mse/10
print("output_mse:", output_mse)