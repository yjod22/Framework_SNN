
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     test.py
#
###############################################################################
#  Description:
#  
#  (For a detailed description look at the object description in the UML model)
#  
###############################################################################
# History
################################################################################
# File:		   holayer.py, test.py
# Version:     5.4 (SCR_V5.3-3)
# Author/Date: Junseok Oh / 2018-11-20
# Change:      Change the order of weights in the convolution operation 
# Cause:       The order of weights in Keras is different with SNN
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.4
# Author/Date: Junseok Oh / 2018-11-14
# Change:      Change the dimension of the weigts from 3D to 4D
#			   Remove test case 1, 2, and 4
# Cause:       (SCR_V5.3-1)
# Initiator:   Junseok Oh
################################################################################
# File:		   hybrid_cnn_passau.py, large_nn_test.py, test.py
# Version:     5.4 (SCR_V5.3-1)
# Author/Date: Junseok Oh / 2018-11-14
# Change:      Use conventional binary function for the 2nd dense layer
# Cause:       The results of 1st dense layer isn't in the range between -1 and +1
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.2 (SCR_V5.1-3)
# Author/Date: Junseok Oh / 2018-10-28
# Change:      Change name of numClasses to numOutputClasses
#			   Create new variable numInputClasses and flagFullyConnected
#			   If the layer has been connected, 
#			   then, it generates the SN, reshapes the format,
#				     and every inputClasses are forwarded as the inputs
# Cause:       Multiple Fully Connected layers should be possible
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.2 (SCR_V5.1-1)
# Author/Date: Junseok Oh / 2018-10-28
# Change:      Place ActivationFunc at HOActivation class
# Cause:       Fully Connected layer also needs activation functions
# Initiator:   Florian Neugebauer
################################################################################
# Version:     5.1
# Author/Date: Junseok Oh / 2018-10-02
# Change:      Add more test cases
# Cause:       (SCR_V5.0-1)
# Initiator:   Junseok Oh
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
# Version:     2.1 (SCR_V2.0-1)
# Author/Date: Junseok Oh / 2018-07-26
# Change:	   Simplify the structure of HOMaxPoolingExact
#              Change the way of bitwise-OR operation using built-in function
# Cause:       Speed-up the simulation time in the Exact Stochastic MaxPooling
# Initiator:   Florian Neugebauer
################################################################################
# Version:     2.0 (SCR_V1.2-1)
# Author/Date: Junseok Oh / 2018-07-17
# Change:	   Create the new HOMaxPoolingExact class
#              Modify HoLayer class's parameter and Add SelectCircuit method
#              Add more test cases
# Cause:       Implementing Object-Oriented Exact Stochastic MaxPooling
# Initiator:   Florian Neugebauer
################################################################################
# Version:     1.2
# Author/Date: Junseok Oh / 2018-07-06
# Change:      Add more test cases
# Cause:       (SCR_V1.1-1, SCR_V1.1-2)
# Initiator:   Junseok Oh
################################################################################
# Version:     1.2 (SCR_V1.1-2)
# Author/Date: Junseok Oh / 2018-07-06
# Change:	   Change the conditions of the iteration in Activation
#			   so that it doesn't access an invalid index of input Matrix
# Cause:       Bug that it failed to Activate when stride is set to 1
# Initiator:   Junseok Oh
################################################################################
# Version:     1.2 (SCR_V1.1-1)
# Author/Date: Junseok Oh / 2018-07-06
# Change:      Create the only one object of HOMaxPooling at the initial phase
#			   Set the new Stochastic Numbers over the iteration
#			   Create SetListSN in HOMaxPooling class
# Cause:       Improve Activation performance
# Initiator:   Junseok Oh
################################################################################ 
# Version:     1.1
# Author/Date: Junseok Oh / 2018-07-05
# Change:      Add more test cases 
# Cause:       Bug fix in Activation
# Initiator:   Junseok Oh
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
from functools import reduce
import operator

class HOLayer(object):
    def __call__(self, inputs, weights, bias, numClasses, denseWeights, denseBias, **kwargs):
        output = self.Call(inputs, weights, bias, numClasses, denseWeights, denseBias, **kwargs)
        return output

    def Call(self, inputs, weights, bias, numClasses, denseWeights, denseBias, **kwargs):
        #This is where the layer's logic lives.
        return inputs

    def SetLayerID(self, x):
        self.layerID = x

    def GetLayerID(self):
        return self.layerID

class HOModel(object):
    def __init__(self, inputMatrix):
        # Calibration values
        self.padding = 0

        # parameter setting
        self.numInputPlanes = 1
        self.numOutputPlanes = 1
        self.inputWidth = int(inputMatrix.size / inputMatrix[0].size)
        self.snLength = int(inputMatrix[0][0].size)

        self.filterSize = 0
        self.stride = 0
        self.numOutputClasses = 1
        self.layerID = 0
        self.outputWidth = 0
        self.cntLayer = 0

        # Initialized snippedMatrix
        self.snippedMatrix = 0

        # Initialized localResult
        self.localResult = np.full(self.snLength, False)

        # Initialized outputMatrix
        self.outputMatrix = 0

        # Initialized copiedMatrix
        self.copiedMatrix = 0
        self.SetCopiedMatrix(self.numInputPlanes, self.inputWidth, self.snLength)

        # Copy inputMatrix into the multi-dimensional copiedMatrix
        #inputMatrix.reshape(inputMatrix.shape + (1,))
        self.copiedMatrix[:] = inputMatrix.reshape(1, 1, self.inputWidth, self.inputWidth, self.snLength)[:]

        # Initialized weights and bias
        self.weightsSN = [0]
        self.biasSN = [0]
        self.denseWeightsSN = [0]
        self.denseBiasSN = [0]

        # Set the flag of Fully-connected as zero (i.e. layers are not yet fully connected)
        self.flagFullyConnected = 0
        self.numInputClasses = 0

    def SetCopiedMatrix(self, PAR_numInputPlanes, PAR_inputWidth, PAR_snLength):
        self.copiedMatrix = np.full((PAR_numInputPlanes, PAR_inputWidth, PAR_inputWidth, PAR_snLength), False)

    def SetInputWidth(self, PAR_inputWidth):
        self.inputWidth = PAR_inputWidth

    def SetOutputWidth(self, PAR_filterSize, PAR_stride):
        self.outputWidth = int(1 + int( (self.inputWidth - PAR_filterSize + 2 * self.padding) / PAR_stride ) )

    def SetSnippedMatrix(self, PAR_filterSize, PAR_snLength):
        self.snippedMatrix = np.full((PAR_filterSize, PAR_filterSize, PAR_snLength), False)

    def SetOutputMatrix(self, PAR_outputWidth, PAR_snLength):
        # The case of Fully connected layer
        if (self.layerID == "FullyConnected"):
            self.outputMatrix = np.zeros((1, self.numOutputClasses))
        # The case of Convolution or MaxPooling layer
        else:
            self.outputMatrix = np.full((self.numOutputPlanes, PAR_outputWidth, PAR_outputWidth, PAR_snLength), False)

    def SetNumInputPlanes(self, x):
        self.numInputPlanes = x

    def SetNumOutputPlanes(self, x):
        self.numOutputPlanes = x

    def SetWeights(self, x):
        self.weightsSN = x

    def SetBias(self, x):
        self.biasSN = x

    def SetDenseWeights(self, x):
        self.denseWeightsSN = x

    def SetDenseBias(self, x):
        self.denseBiasSN = x

    def CopyMatrix(self):
        self.copiedMatrix[:] = self.outputMatrix[:]

    def IncrementCntLayer(self):
        self.cntLayer = self.cntLayer + 1

    def GetCntLayer(self):
        return self.cntLayer

    def GetOutputMatrix(self):
        return self.outputMatrix

    def Activation(self, holayer, **kwargs):
        # Initialize index of output Row and Column
        outputRow = 0
        outputCol = 0

        # Set stride and the number of classes
        for key in kwargs:
            if (key == "stride"):
                self.stride = kwargs[key]
            elif (key == "num_classes"):
                self.numOutputClasses = kwargs[key]

        self.layerID = holayer.GetLayerID()

        # If it is the first activation of the model, then skip the followings
		# Otherwise, it will change the current output Matrix as the input Matrix
        self.IncrementCntLayer()
        if(self.GetCntLayer() > 1):

            # If the layer has been fully connected, then it generates the SN and reshapes the format
            # e.g. (1, 10) -> SN generation -> (10, 1024) -> reshape -> (10, 1, 1, 1024)
            if(self.flagFullyConnected == 1):
                tt = self.outputMatrix.size
                tt1 = self.outputMatrix[0].size
                tt2 = self.outputMatrix[0][0].size
                self.numInputClasses = int(self.outputMatrix[0].size / self.outputMatrix[0][0].size)
                resMatrix = np.full((self.numInputClasses, self.snLength), False)
                for i in range(self.numInputClasses):
                    resMatrix[i] = self.CreateSN(self.outputMatrix[0, i], self.snLength)
                self.SetCopiedMatrix(self.numInputClasses, 1, self.snLength)
                self.copiedMatrix = resMatrix.reshape(self.numInputClasses, 1, 1, self.snLength)

            else:
                self.SetNumInputPlanes(int(self.outputMatrix.size / self.outputMatrix[0].size))
                self.SetInputWidth(self.outputWidth)
                self.SetCopiedMatrix(self.numInputPlanes, self.inputWidth, self.snLength)
                self.CopyMatrix()

        # Depending on the filter size and stride of the layer,
		# it determines the OutputMatrix paramenters
        # In the case of Fully connected layer, Determine filterSize and stride from inputWidth
        if(self.layerID == "FullyConnected"):
            self.filterSize = self.inputWidth
            self.stride = self.filterSize
        # In the case of Convolution or MaxPooling layer, Determine filterSize from the object
        else:
            self.filterSize = holayer.GetFilterSize()
        self.SetOutputWidth(self.filterSize, self.stride)
        self.SetSnippedMatrix(self.filterSize, self.snLength)
        self.SetOutputMatrix(self.outputWidth, self.snLength)


        # Iterate over the Planes
        for i in range(self.numOutputPlanes):

            # In the case of Fully connected layer, DenseFunc is called only once
            if ( (self.layerID == "FullyConnected") and (i > 0)):
                break

            # Iterate over the row of the input Matrix by the stride
            for row in range(0, self.inputWidth, self.stride):

                # Set the row index of the output Matrix
                outputRow = int(row / self.stride)

                # If it accesses invalid index of the input Matrix's row, then it skips the following logic
                if ((row + self.filterSize) > self.inputWidth):
                    break

                else:
                    # Iterate over the column of the input Matrix by the stride
                    for col in range(0, self.inputWidth, self.stride):

                        # Set the column index of the output Matrix
                        outputCol = int(col / self.stride)

                        # If it accesses invalid index of the input Matrix's column, then it skips the following logic
                        if ((col + self.filterSize) > self.inputWidth):
                            break

                        else:
                            # It calls the implementation of Convolution, MaxPooling, or Dense
                            self.localResult = holayer(self.Snip(i, row, col),
                                                       self.weightsSN[i], self.biasSN[i],
                                                       self.numOutputClasses, self.denseWeightsSN, self.denseBiasSN)

                            # In the case of Fully connected layer, Iterate over the Classes
                            for j in range(self.numOutputClasses):
                                # Fill the result of Convolution or MaxPooling into the output Matrix
                                self.FillOutput(j, i, outputRow, outputCol)

    def Snip(self, ithPlane, row, col):
        # In the case of MaxPooling layer
        if (self.layerID == "MaxPooling"):
            self.snippedMatrix = self.copiedMatrix[ithPlane, row:row+self.filterSize, col:col+self.filterSize]
        # In the case of Convolution or Fully connected layer, whole planes are forwarded as the inputs
        else:
            if(self.flagFullyConnected == 0):
                self.snippedMatrix = self.copiedMatrix[0:self.numInputPlanes, row:row+self.filterSize, col:col+self.filterSize]
            # If the layer has been fully connected, then every inputClasses are forwarded as the inputs
            if(self.flagFullyConnected == 1):
                self.snippedMatrix = self.copiedMatrix[0:self.numInputClasses, row:row+self.filterSize, col:col+self.filterSize]

        return self.snippedMatrix

    def FillOutput(self, jthClass, ithPlane, row, col):
        # In the case of Fully connected layer
        if (self.layerID == "FullyConnected"):
            self.outputMatrix[0, jthClass] = self.localResult[0][jthClass]
            self.flagFullyConnected = 1
        # In the case of Convolution or MaxPooling layer
        else:
            self.outputMatrix[ithPlane, row, col] = self.localResult[0]

    def CreateSN(self, x, length):
        """create bipolar SN by comparing random vector elementwise to SN value x"""
        # rand = np.random.rand(length)*2.0 - 1.0
        # x_SN = np.less(rand, x)
        large = np.random.rand(1)
        x_SN = np.full(length, False)
        if large:
            for i in range(int(np.ceil(((x + 1) / 2) * length))):
                x_SN[i] = True
        else:
            for i in range(int(np.floor(((x + 1) / 2) * length))):
                x_SN[i] = True
        np.random.shuffle(x_SN)
        return x_SN
class HOActivation(HOLayer):
    def ActivationFuncReluSN(self, x):
        '''input x is a stochastic number'''
        if sum(x) > len(x) / 2:
            return x
        else:
            return self.zeroSN

    def ActivationFuncRelu(selfs, x):
        '''input x is not a stochastic number'''
        if (x <= 0):
            return 0
        else:
            return x

    def ActiavtionFuncTanhSN(self, x):
        """activation function for stochastic NN. FSM with 8 states"""
        # the number of states
        numState = 8
        # starting state
        state = 3
        out = np.full(len(x), True, dtype=bool)
        for j in range(len(x)):
            '''
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

            '''
            # Determine the output depending on the current state
            if ((0 <= state) & (state < int(numState/2))):
                out[j] = 0
            else:
                out[j] = 1
            # input is True -> increase state
            if x[j] & (state < numState):
                state = state + 1
            # input is False -> decrease state
            elif np.logical_not(x[j]) & (state > 0):
                state = state - 1
            # No update at the start or end of the state
        print("after stanh")
        print(out)

        return out

class HOMaxPooling(HOLayer):
    def Call(self, inputs, weights, bias, numClasses, denseWeights, denseBias, **kwargs):
        output = self.PoolingFunc(inputs)
        return output

    def PoolingFunc(self, inputs):
        raise NotImplementedError

class HOConv(HOActivation):
    def Call(self, inputs, weights, bias, numClasses, denseWeights, denseBias, **kwargs):
        output = self.ConvFunc(inputs, weights, bias)
        return output

    def ConvFunc(self, inputs, weights, bias):
        raise NotImplementedError

class HOConn(HOActivation):
    def Call(self, inputs, weights, bias, numClasses, denseWeights, denseBias, **kwargs):
        output = self.DenseFunc(inputs, numClasses, denseWeights, denseBias)
        return output

    def DenseFunc(self, inputs, numClasses, denseWeights, denseBias):
        raise NotImplementedError

class HOConnected(HOConn):
    def __init__(self, PAR_SnLength, **kwargs):
        self.SetLayerID("FullyConnected")
        self.snLength = PAR_SnLength
        self.dense_output_SN = [0]

        # Select the way of transforming stochastic number to integer
        for key in kwargs:
            if (key == "stochToInt"):
                self.stochToInt = kwargs[key]
            if (key == "activationFunc"):
                self.activationFunc = kwargs[key]

    def DenseFunc(self, inputs, numClasses, denseWeights, denseBias):
        # Flatten the inputs
        t = inputs.size
        t2 = inputs[0].size
        t3= inputs[0][0].size
        t4 = inputs[0][0][0].size

        y = denseWeights.size
        y2 = denseWeights[0].size
        y3 = denseWeights[0][0].size
        y4 = denseWeights[0][0][0].size

        sizeTensor = int (inputs.size / inputs[0][0][0].size) # The total number of elements in the whole layer
        denseInputs = inputs.reshape((1, sizeTensor, self.snLength))

        z = denseInputs.size
        z2 = denseInputs[0].size
        z3 = denseInputs[0][0].size
        z4 = denseInputs[0][0][0].size

        self.dense_output_SN = np.full((numClasses, sizeTensor, self.snLength), False)
        #self.dense_outputN = np.zeros((1, numClasses))
        self.dense_output = np.zeros((1, numClasses))

        # PRODUCT in the inner product operations
        for i in range(sizeTensor):
            for j in range(numClasses):
                self.dense_output_SN[j, i] = np.logical_not(np.logical_xor(denseInputs[0, i], denseWeights[i, j]))

        # ADD in the inner product operations
        if (self.stochToInt == "Normal"):
            for i in range(numClasses):
                for j in range(sizeTensor):
                    self.dense_output[0, i] = self.dense_output[0, i] + self.StochToInt(self.dense_output_SN[i, j])

        elif (self.stochToInt == "APC"):
            for i in range(numClasses):
                self.dense_output[0, i] = self.APC(self.dense_output_SN[i], self.snLength, sizeTensor)

        #print("Normal")
        #print(self.dense_outputN[0, 0])
        #print("APC")
        #print(self.dense_output[0, 0])

        # Biasing
        self.dense_output = self.dense_output + denseBias

        # Activation function
        if (self.activationFunc == "Relu"):
            for i in range(len(self.dense_output[0])):
                self.dense_output[0][i] = self.ActivationFuncRelu(self.dense_output[0][i])

        elif (self.activationFunc == "Tanh"):
            '''Not yet implemented'''
            #self.dense_output = self.ActiavtionFuncTanh(self.dense_output)
            pass

        return self.dense_output

    def APC(self, x, snLength, sizeTensor):
        '''
        print("x")
        print(x[0:16])
        print(x[0])
        SN1 = x[0]
        SN2 = x[1]
        SN3 = x[2]
        SN4 = x[3]
        SN5 = x[4]
        SN6 = x[5]
        SN7 = x[6]
        SN8 = x[7]
        SN9 = x[8]
        SN10 = x[9]
        SN11 = x[10]
        SN12 = x[11]
        SN13 = x[12]
        SN14 = x[13]
        SN15 = x[14]
        SN16 = x[15]


        SN17 = x[16]
        SN18 = x[17]
        SN19 = x[18]
        SN20 = x[19]
        SN21 = x[20]
        SN22 = x[21]
        SN23 = x[22]
        SN24 = x[23]
        SN25 = x[24]
        SN26 = x[25]
        SN27 = x[26]
        SN28 = x[27]
        SN29 = x[28]
        SN30 = x[29]
        SN31 = x[30]
        SN32 = x[31]

        SN33 = x[32]
        SN34 = x[33]
        SN35 = x[34]
        SN36 = x[35]
        SN37 = x[36]
        SN38 = x[37]
        SN39 = x[38]
        SN40 = x[39]
        SN41 = x[40]
        SN42 = x[41]
        SN43 = x[42]
        SN44 = x[43]
        SN45 = x[44]
        SN46 = x[45]
        SN47 = x[46]
        SN48 = x[47]
        '''
        numAPC = int(sizeTensor / 16)
        sum = 0
        for i in range(numAPC):
            ithSum = 0
            for j in range(snLength):
                # AND, OR gates
                a = (x[0+16*i, j] | x[1+16*i, j])
                b = (x[2+16*i, j] & x[3+16*i, j])
                c = (x[4+16*i, j] | x[5+16*i, j])
                d = (x[6+16*i, j] & x[7+16*i, j])
                e = (x[8+16*i, j] | x[9+16*i, j])
                f = (x[10+16*i, j] & x[11+16*i, j])
                z2 = (x[12+16*i, j] | x[13+16*i, j])
                t0 = (x[14+16*i, j] & x[15+16*i, j])

                # Full Adder 1 (Carry:x1, Sum:x2)
                x1 = ((a & b) | (b & c) | (c & a))
                x2 = ((a ^ b) ^ c)

                # Full Adder 2 (Carry:y1, Sum:y2)
                y1 = ((d & e) | (e & f) | (f & d))
                y2 = ((d ^ e) ^ f)

                # Full Adder 3 (Carry:z1, Sum:t1)
                z1 = ((x2 & y2) | (y2 & z2) | (z2 & x2))
                t1 = ((x2 ^ y2) ^ z2)

                # Full Adder 4 (Carry:t3, Sum:t2)
                t3 = ((x1 & y1) | (y1 & z1) | (z1 & x1))
                t2 = ((x1 ^ y1) ^ z1)

                # Represent in the binary format
                ithSum = ithSum + 8*t3 + 4*t2 + 2*t1 + 2*t0
                #print(sum)
                #tttt=1
            ithIntSum = ithSum/snLength*2 - 16
            sum = sum + ithIntSum
            #xxxx=1

        #print("sum")
        #print(sum)
        #zz = 1
        return sum

    def StochToInt(self, x):
        """convert bipolar stochastic number to integer"""
        return (sum(x) / len(x)) * 2.0 - 1.0

class HOConvolution(HOConv):
    def __init__(self, PAR_Row, PAR_Col, PAR_SnLength, **kwargs):
        # parameter setting
        self.SetLayerID("Convolution")
        self.row = PAR_Row
        self.col = PAR_Col
        self.matrixSize = PAR_Row * PAR_Col
        self.snLength = PAR_SnLength

        # Initialize the Convolution output
        self.listProductSN = 0
        self.convOutput = np.full((1, self.snLength), False)

        # Create the Stochastic Number Zero
        self.zeroSN = self.CreateSN(0, self.snLength)

        # Select the activation function to use
        for key in kwargs:
            if (key == "activationFunc"):
                    self.activationFunc = kwargs[key]

    def GetFilterSize(self):
        return self.row

    def ConvFunc(self, inputs, weights, bias):
        # Initialize the Convolution output
        numInputPlanes = int (inputs.size / inputs[0].size) # The number of planes in the inputs
        self.listProductSN = np.full((numInputPlanes*self.matrixSize+1, self.snLength), False)

        for k in range(numInputPlanes):
            for i in range(self.row):
                for j in range(self.col):
                    self.listProductSN[k*self.matrixSize+i*self.col+j] = np.logical_not(np.logical_xor(weights[k, self.row-1-i, self.col-1-j],
                                                                                                       inputs[k, i, j]))
        self.listProductSN[numInputPlanes*self.matrixSize] = bias

        for i in range(self.snLength):
            r = np.floor(np.random.rand(1)[0]*(numInputPlanes*self.matrixSize+1))
            self.convOutput[0][i] = self.listProductSN[int(r), i]

        if(self.activationFunc == "Relu"):
            self.convOutput[0] = self.ActivationFuncReluSN(self.convOutput[0])
        elif(self.activationFunc == "Tanh"):
            self.convOutput[0] = self.ActiavtionFuncTanhSN(self.convOutput[0])

        return self.convOutput

    def CreateSN(self, x, length):
        """create bipolar SN by comparing random vector elementwise to SN value x"""
        # rand = np.random.rand(length)*2.0 - 1.0
        # x_SN = np.less(rand, x)
        large = np.random.rand(1)
        x_SN = np.full(length, False)
        if large:
            for i in range(int(np.ceil(((x + 1) / 2) * length))):
                x_SN[i] = True
        else:
            for i in range(int(np.floor(((x + 1) / 2) * length))):
                x_SN[i] = True
        np.random.shuffle(x_SN)
        return x_SN

class HOMaxPoolingExact(HOMaxPooling):
    def __init__(self, PAR_Row, PAR_Col, PAR_SnLength):
        # parameter setting
        self.SetLayerID("MaxPooling")
        self.row = PAR_Row
        self.col = PAR_Col
        self.matrixSize = PAR_Row * PAR_Col
        self.snLength = PAR_SnLength

        #andGates
        self.listOutput = []
        for i in range(self.matrixSize):
            self.listOutput.append(0)

        #orGate
        self.snOut = 0

        #CounterExtended
        self.listCnt = []
        for i in range(self.matrixSize):
            self.listCnt.append(0)

        # Extract Stochastic Number from the matrix x.
        self.listSN = []
        # self.SetListSN(x)

        # Initialize the MaxPooling output
        self.maxOutput = np.full((1, PAR_SnLength), False)

    def GetFilterSize(self):
        return self.row

    def SetListSN(self, x):
        # Delete all elements in listSN
        self.listSN.clear()

        # Extract Stochastic Number from the matrix x.
        for i in range(self.row):
            for j in range(self.col):
                self.listSN.append(x[i, j])

    def CleanUpRegister(self):
        self.listCnt = []
        for i in range(self.matrixSize):
            self.listCnt.append(0)

    def PoolingFunc(self, inputs):
        self.CleanUpRegister()
        self.SetListSN(inputs)

        for i in range(self.snLength):
            for j in range(self.matrixSize):
                self.listOutput[j] = self.listSN[j][i] & (self.listCnt[j] == 0)

            self.maxOutput[0][i] = any(self.listOutput)

            for j in range(self.matrixSize):
                inc = ~(self.listSN[j][i]) & self.maxOutput[0][i]

                temp = self.listOutput[j]
                self.listOutput[j] = False

                dec = self.listSN[j][i] & (self.listCnt[j] != 0) & ~(any(self.listOutput))
                self.listOutput[j] = temp

                self.listCnt[j] = self.listCnt[j] + inc - dec

        return self.maxOutput

class HOMaxPoolingAprox(HOMaxPooling):
    def __init__(self, PAR_Row, PAR_Col, PAR_SnLength, PAR_step):
        # parameter setting
        self.SetLayerID("MaxPooling")
        self.row = PAR_Row
        self.col = PAR_Col
        self.matrixSize = PAR_Row * PAR_Col
        self.ithSN = np.random.rand(1) * (self.matrixSize-1)
        self.ithSN = int(np.ceil(self.ithSN[0]))
        self.snLength = PAR_SnLength
        self.numPartialSN = int (PAR_SnLength / PAR_step)
        self.step = PAR_step

        # Create objects
        self.mux = Mux()
        self.comparator = Comparator()
        self.listCounter = []
        for i in range(self.matrixSize):
            counter = Counter()
            self.listCounter.append(counter)

        # Extract Stochastic Number from the matrix x.
        self.listSN = []
        #self.SetListSN(x)

        # Initialize the MaxPooling output
        self.maxOutput = np.full((1, PAR_SnLength), False)

    def GetFilterSize(self):
        return self.row

    def SetListSN(self, x):
        # Delete all elements in listSN
        self.listSN.clear()

        # Extract Stochastic Number from the matrix x.
        for i in range(self.row):
            for j in range(self.col):
                self.listSN.append(x[i, j])

    def PoolingFunc(self, inputs):
        self.SetListSN(inputs)

        # Iterate over the c-bits of the Stochastic number
        for nthPartialSN in range(self.numPartialSN-1):

            # Run the counter objects for counting
            for i in range(self.matrixSize):
                self.listCounter[i].Count(self.listSN[i], (nthPartialSN)*self.step, (nthPartialSN+1)*self.step)

            # Find a maximum value
            self.comparator.FindMax(self.listCounter)

            # Select SN based on the maximum value
            self.mux.Select(self.listSN, self.maxOutput, self.comparator, self.listCounter, (nthPartialSN+1)*self.step, (nthPartialSN+2)*self.step)

        # Fill in the first c-bits of the MaxPooling output
        self.mux.SelectFirstBit(self.listSN, self.maxOutput, self.ithSN, self.step)

        return self.maxOutput

class Mux(object):
    def __init__(self):
        pass

    def Select(self, listSN, maxOutput, comparator, listCounter, start, end):
        for i in range(len(listSN)):
            localCnt = listCounter[i].GetCnt()
            if(localCnt == comparator.GetMaxima() ):
                maxOutput[0][start:end] = listSN[i][start:end]
            else:
                pass


    def SelectFirstBit(self, listSN, maxOutput, ithSN, step):
        maxOutput[0][0:step] = listSN[ithSN][0:step]


class Comparator(object):
    def __init__(self):
        self.maxima = 0

    def GetMaxima(self):
        return self.maxima

    def SetMaxima(self, x):
        self.maxima = x

    def FindMax(self, listCounter):
        listMaxCount = []
        for i in range(len(listCounter)):
            listMaxCount.append(listCounter[i].GetCnt())
        self.maxima = max(listMaxCount[:])


class Counter(object):
    def __init__(self):
        self.cnt = 0

    def GetCnt(self):
        return self.cnt

    def SetCnt(self, x):
        self.cnt = x

    def Count(self, sn, start, end):
        self.cnt = sum(sn[start:end])




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
    result = stanh(result)
    #result = srelu(result)
    return result


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



def aprox_smax2(x):
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

    step = 3

    for l in range(int(len(SN1)/step) - 1):
        t1= (step * l)
        t2= ((l + 1) * step )

        counter1 = sum(SN1[(step * l):((l + 1) * step )])
        counter2 = sum(SN2[(step * l):((l + 1) * step )])
        counter3 = sum(SN3[(step * l):((l + 1) * step )])
        counter4 = sum(SN4[(step * l):((l + 1) * step )])

        t3= (step * (l + 1))
        t4= (((l + 1) + 1) * step )

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
    #print(SN_out)
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

        pass

    print(SN_out)
    return SN_out



length = 10

#test set 03
print("test set 03")
SN_input_matrix = np.full((6, 6, length), False)
SN_input_matrix[0:4, 0:3] = [[[True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True]],
                            [[True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True]],
                            [[True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True]],
                            [[True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True]]
                           ]
print("SN_input_matrix")
print(SN_input_matrix)

weight_SNs = np.full((3, 1, 3, 3, length), True)
print("weight_SNs")
print(weight_SNs)

weight2_SNs = np.full((4, 3, 1, 1, length), True)
print("weight2_SNs")
print(weight2_SNs)

bias_SNs = np.full((3, length), False)
bias_SNs[0] = [True, True, True, True, True, True, True, True, True, True]
bias_SNs[1] = [True, True, True, True, True, True, True, True, True, True]
bias_SNs[2] = [True, True, True, True, True, True, True, True, True, True]
print("bias_SNs")
print(bias_SNs)

bias2_SNs = np.full((4, length), False)
bias2_SNs[0] = [True, True, True, True, True, True, True, True, True, True]
bias2_SNs[1] = [True, True, True, True, True, True, True, True, True, True]
bias2_SNs[2] = [True, True, True, True, True, True, True, True, True, True]
bias2_SNs[3] = [True, True, True, True, True, True, True, True, True, True]
print("bias2_SNs")
print(bias2_SNs)

# calculate output of convolutional layer
#for i in range(3):
#    for j in range(4):
#        for k in range(4):
#            conv_layer_output[i, j, k] = neuron(weight_SNs[i], SN_input_matrix[j:(j + 3), k:(k + 3)], bias_SNs[i])
#print("conv_layer_output")
#print(conv_layer_output)
layer1 = HOModel(SN_input_matrix)

layer1.SetNumOutputPlanes(3)
layer1.SetWeights(weight_SNs)
layer1.SetBias(bias_SNs)
layer1.Activation(HOConvolution(3, 3, length, activationFunc="Relu"), stride=1)
print("Convolution done")
print(layer1.GetOutputMatrix())

layer1.Activation(HOMaxPoolingExact(2, 2, length), stride=2)
print("max pool done")
print(layer1.GetOutputMatrix())

layer1.SetNumOutputPlanes(4)
layer1.SetWeights(weight2_SNs)
layer1.SetBias(bias2_SNs)
layer1.Activation(HOConvolution(1, 1, length, activationFunc="Relu"), stride=1)
print("Convolution2 done")
print(layer1.GetOutputMatrix())

layer1.Activation(HOMaxPoolingExact(2, 2, length), stride=1)
print("max pool2 done")
print(layer1.GetOutputMatrix())

dense_weight_SNs = np.full((4, 2, length), True)
#print(dense_weight_SNs)
#SN_input_matrix = np.full((6, 6, length), False)
dense_weight_SNs[0:4, 0:2] = [[[True, True, True, True, True, True, True, True, True, True],
                            [False, False, False, False, False, False, False, False, False, False]]
    ,
                            [[True, True, True, True, True, True, True, True, True, True],
                            [False, False, False, False, False, False, False, False, False, False]]
    ,
                            [[True, True, True, True, True, True, True, True, True, True],
                            [False, False, False, False, False, False, False, False, False, False]]
    ,
                            [[True, True, True, True, True, True, True, True, True, True],
                            [False, False, False, False, False, False, False, False, False, False]]
                           ]

#print(dense_weight_SNs)
dense_biases = [4, 4]
layer1.SetDenseWeights(dense_weight_SNs)
layer1.SetDenseBias(dense_biases)
layer1.Activation(HOConnected(length, stochToInt="Normal", activationFunc="Relu"), num_classes=2)

x = layer1.GetOutputMatrix()
#dense_input_SN = x.reshape((1, 48, length)) # 4*4*3
print("fully connected")
#print(dense_input_SN)
print(x)



