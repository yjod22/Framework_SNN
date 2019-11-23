import keras
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from keras import regularizers
from keras import backend as K
from keras.callbacks import Callback
from keras.datasets import mnist
import numpy as np
from snn.hoModel import HOModel
from snn.hoLayer import HOMaxPoolingExact, HOMaxPoolingAprox, HOConvolution, HOConnected
from snn.hoUtils import HOUtils
from snn.bnLayer import BNModel
from keras.utils.generic_utils import get_custom_objects
import global_variables
import WeightScale

np.set_printoptions(threshold=np.inf)

batch_size = 128
num_classes = 10
epochs = 1
#cntEpochs = 0
input_shape = 0

# Define the global variables
global_variables.DefineGlobalVariables()


def first_layer_activation(x):
    return K.tanh(x)


def load_data():
    num_classes = 10
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return (x_train, y_train, x_test, y_test, input_shape)


get_custom_objects().update({'first_layer_activation': Activation(first_layer_activation)})
(x_train, y_train, x_test, y_test, input_shape) = load_data()

#input_shape = x_train.shape[1:]
print(input_shape)


global_variables.bnModel = BNModel(8)
global_variables.bnModel.SetId(1)
global_variables.bnModel[0] = Conv2D(3, kernel_size=(5, 5), input_shape=input_shape, use_bias=False, kernel_regularizer=regularizers.l1(0.001))
global_variables.bnModel[1] = Activation(first_layer_activation)
global_variables.bnModel[2] = MaxPooling2D(pool_size=(2, 2))
global_variables.bnModel[3] = Flatten()
global_variables.bnModel[4] = Dense(100)
global_variables.bnModel[5] = Activation('relu')
global_variables.bnModel[6] = Dense(num_classes)
global_variables.bnModel[7] = Activation('softmax')
global_variables.bnModel.LoadLayers()
global_variables.bnModel.Compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#global_variables.bnModel.Fit(x_train, y_train,
#                             batch_size=batch_size,
#                             epochs=epochs,
#                             verbose=1,
#                             callbacks=[WeightScale()],
#                             validation_data=(x_test, y_test))
global_variables.bnModel.Load_weights('../results/#Epoch1_weights_of_1st_model_network_test.h5')
global_variables.bnModel.Evaluate(x_test, y_test, verbose=1, indexModel=1)
global_variables.bnModel.OptimizeNetwork('network_test', '../results/#Epoch1_weights_of_1st_model_network_test.h5',
                                         '../results/#Epoch1_weights_of_1st_model_network_test.h5',
                                         WeightScale,
                                         cntIter=1,
                                         tupleLayer=(1, ),
                                         x_train=x_train,
                                         y_train=y_train,
                                         x_test=x_test,
                                         y_test=y_test,
                                         epochs=1,
                                         batch_size=batch_size)

# Get the layer models from bnModel
layer1model = global_variables.bnModel[0]
layer2model = global_variables.bnModel[1]
layer3model = global_variables.bnModel[2]
layer4model = global_variables.bnModel[3]
layer5model = global_variables.bnModel[4]
layer6model = global_variables.bnModel[5]
layer7model = global_variables.bnModel[6]
layer8model = global_variables.bnModel[7]

kBits = 10
length = 2**kBits

ut = HOUtils(kBits=kBits)
model = global_variables.bnModel.GetModel()
weight_1_SNs, bias_1_SNs, listIndex1= ut.GetConvolutionLayerWeightsBiasesSN(model, 1, Adaptive="True")

dense_1_weight_SNs= ut.GetConnectedLayerWeightsSN(model, 5)
dense_1_biases= ut.GetConnectedLayerBiases(model, 5)
dense_2_weights= ut.GetConnectedLayerWeights(model, 7)
dense_2_biases= ut.GetConnectedLayerBiases(model, 7)


correct_predictions = 0
output_mse = 0
SN_input_matrix = np.full((28, 28, length), False)
for test_index in range(10000):
    print("test image:", test_index)
    x = x_test[test_index]
    for j in range(28):
        for k in range(28):
            SN_input_matrix[j, k] = ut.CreateSN(x[0, j, k])

    hoModel = HOModel(SN_input_matrix, kBits=kBits)
    hoModel.SetNumOutputPlanes(3)
    hoModel.SetWeights(weight_1_SNs)
    hoModel.SetZeroBias(3)
    hoModel.SetListIndex(listIndex1)
    hoConvLayer = HOConvolution(5, 5, kBits=kBits, use_bias = "False", baseMode = "Mux", activationFunc="STanh")
    hoModel.Activation(hoConvLayer, stride=1)

    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    #ut.SaveInTxtFormat('../results/network_test_conv1', test_index,
    #                   hoModel.GetOutputMatrix(), 3, 24, 24,
    #                   layer2model, x_test)
    print(str(test_index + 1) + ' conv 1 layer results saved in txt format')
    ###################################################################################################################

    hoMaxLayer = HOMaxPoolingExact(2, 2, kBits=kBits)
    hoModel.Activation(hoMaxLayer, stride=2)

    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    #ut.SaveInTxtFormat('../results/network_test_maxpool', test_index,
    #                   hoModel.GetOutputMatrix(), 3, 12, 12,
    #                   layer3model, x_test)
    print(str(test_index + 1) + ' maxpool layer results saved in txt format')
    ###################################################################################################################

    hoModel.SetNumOutputPlanes(1)
    hoModel.SetDenseWeights(dense_1_weight_SNs)
    hoModel.SetDenseBias(dense_1_biases)
    hoDenseLayer = HOConnected(kBits=kBits, use_bias="True", stochToInt="APC", activationFunc="Relu")
    hoModel.Activation(hoDenseLayer, num_classes=100)

    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    #dense_output = hoModel.GetOutputMatrix()
    #print("dense 1 output from Binary NN with ReLU")
    #BNN_prediction = layer6model.predict(np.asarray([x_test[test_index]]))
    #print(BNN_prediction)
    #del(BNN_prediction)
    #print("dense 1 output from Stochastic NN with ReLU")
    #print(dense_output)
    ###################################################################################################################

    dense_output = hoModel.GetOutputMatrix()
    dense_output = ut.BinaryConnectedLAyer(100, num_classes, dense_output, dense_2_weights, dense_2_biases)

    ################### For debugging purpose, save the intermidiate results in the local variable ###################
    print("dense 2 output from Binary NN without softmax")
    BNN_prediction = layer7model.predict(np.asarray([x_test[test_index]]))
    print(BNN_prediction)
    del (BNN_prediction)
    print("dense 2 output from Stochastic NN without softmax")
    print(dense_output)
    ###################################################################################################################
    print('dense 2 layer done')

    out_error = 0
    out = layer7model.predict(np.asarray([x_test[test_index]]))
    for i in range(10):
        out_error = out_error + (dense_output[0, i] - out[0, i]) ** 2

    print("Current output_mse:", out_error)
    output_mse = output_mse + out_error

    # softmax activation
    dense_out_exp = [np.exp(i) for i in dense_output]
    exp_sum_out = np.sum(dense_out_exp)
    hybrid_output = [i / exp_sum_out for i in dense_out_exp]
    print('dense done with the softmax activation function')
    print("Labeled output of the dense layer")
    print(np.argmax(y_test[test_index]))
    print("SNN results of dense layer")
    print(np.argmax(hybrid_output))

    if(np.argmax(hybrid_output) == np.argmax(y_test[test_index])):
        correct_predictions = correct_predictions + 1

    print("accuracy:", correct_predictions/(test_index+1))