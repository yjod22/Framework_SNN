from snn.bnLayer import BNModel
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from keras import regularizers
from keras import backend as K
from keras.callbacks import Callback
import keras
from keras.datasets import mnist
from keras.utils.generic_utils import get_custom_objects
import numpy as np
import snn.hoUtils as hoU
import snn.hoModel as hoM
import snn.hoLayer as hoL
import global_variables
import WeightScale


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
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    return (x_train, y_train, x_test, y_test)


get_custom_objects().update({'first_layer_activation': Activation(first_layer_activation)})
(x_train, y_train, x_test, y_test) = load_data()

input_shape = x_train.shape[1:]
print(input_shape)

global_variables.bnModel = BNModel(8)
global_variables.bnModel.SetId(1)
global_variables.bnModel[0] = Conv2D(3, kernel_size=(5, 5), input_shape=input_shape, use_bias=False, kernel_regularizer=regularizers.l1(0.001))
global_variables.bnModel[1] = Activation('first_layer_activation')
global_variables.bnModel[2] = MaxPooling2D(pool_size=(2, 2))
global_variables.bnModel[3] = Flatten()
global_variables.bnModel[4] = Dense(100)
global_variables.bnModel[5] = Activation('relu')
global_variables.bnModel[6] = Dense(10)
global_variables.bnModel[7] = Activation('softmax')
global_variables.bnModel.LoadLayers()
global_variables.bnModel.Compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
global_variables.bnModel.Fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, callbacks=[WeightScale.WeightScale()], validation_data=(x_test, y_test))

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
                                         batch_size=128)

kBits = 10
length = 2**kBits

ut=hoU.HOUtils(kBits=kBits)
model = global_variables.bnModel.GetModel()
weight_1_SNs, bias_1_SNs, listIndex1 = ut.GetConvolutionLayerWeightsBiasesSN(model, 1, Adaptive="True")

#dense_1_weight_SNs= ut.GetConnectedLayerWeightsSN(model, 5)
#dense_1_biases= ut.GetConnectedLayerBiases(model, 5)
dense_1_weight_SNs, dense_1_biases_SNs, listIndexDense = ut.GetConnectedLayerWeightsBiasesSN(model, 5, Adaptive="True")

dense_2_weights= ut.GetConnectedLayerWeights(model, 7)
dense_2_biases= ut.GetConnectedLayerBiases(model, 7)


correct_predictions = 0
SN_input_matrix = np.full((28, 28, length), False)
for i in range(10000):
    print("test image:", i)
    x = x_test[i]
    for j in range(28):
        for k in range(28):
            SN_input_matrix[j, k] = ut.CreateSN(x[0, j, k])

    hoModel = hoM.HOModel(SN_input_matrix, kBits=kBits)
    hoModel.SetNumOutputPlanes(3)
    hoModel.SetWeights(weight_1_SNs)
    hoModel.SetZeroBias(3)
    hoModel.SetListIndex(listIndex1)
    hoConvLayer = hoL.HOConvolution(5, 5, kBits=kBits, use_bias="False", baseMode="Mux", activationFunc="STanh")
    hoModel.Activation(hoConvLayer, stride=1)
    hoMaxLayer = hoL.HOMaxPoolingExact(2, 2, kBits=kBits)
    hoModel.Activation(hoMaxLayer, stride=2)
    hoModel.SetNumOutputPlanes(1)
    hoModel.SetDenseWeights(dense_1_weight_SNs)
    hoModel.SetDenseBias(dense_1_biases_SNs)
    hoModel.SetListIndexDense(listIndexDense)
    hoDenseLayer = hoL.HOConnected(kBits=kBits, use_bias="True", stochToInt="APC", activationFunc="Relu")
    hoModel.Activation(hoDenseLayer, num_classes=100)
    output = hoModel.GetOutputMatrix()
    output = ut.BinaryConnectedLAyer(100, 10, output, dense_2_weights, dense_2_biases)

    if (np.argmax(output) == np.argmax(y_test[i])):
        correct_predictions = correct_predictions + 1

    print("accuracy:", correct_predictions/(i+1))