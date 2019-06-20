#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     holayer.py
#
###############################################################################
#  Description:
#  
#  (For a detailed description look at the object description in the UML model)
#  
###############################################################################
# History
################################################################################
# File:		   holayer.py
# Version:     11.0
# Author/Date: Junseok Oh / 2019-06-20
# Change:      (SCR_V10.0-1): Pre-processing in APCs
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py
# Version:     10.0
# Author/Date: Junseok Oh / 2019-06-17
# Change:      (SCR_V9.0-1): Deploy SC-based ReLU
#              (SCR_V9.0-2): Deploy APCs (8, 16, 25bits)
#              (SCR_V9.0-3): Generate snLookupTableNumAPC
# Cause:       Catch up with the recent new research outcomes
# Initiator:   Junseok Oh
################################################################################
# File:		   holayer.py
# Version:     9.0
# Author/Date: Junseok Oh / 2019-06-07
# Change:      (SCR_V8.0-2): Fix bug of set the state in ActivationFuncTanhSN
#			   (SCR_V8.0-3): develop LUT-based APC
#			   (SCR_V8.0-4): develop 8bit APC
#			   (SCR_V8.0-5): Increase dimension readability by shape
#			   (SCR_V8.0-6): Apply LUT-based techniques in dense layer
# Cause:       Performance improvements
# Initiator:   Junseok Oh
################################################################################
# File:		   holayer.py
# Version:     8.0
# Author/Date: Junseok Oh / 2019-05-23
# Change:      (SCR_V6.4-1): NN Optimization-JSO (Make use of listIndex not to consider zero weights in addition)
#			   (SCR_V6.4-2): Fix bug in the number of states in the ActivationFuncTanhSN
#			   (SCR_V6.4-9): Update Stanh with LUT for adaptive function
#			   (SCR_V6.4-11): Fix bug of bias missing (use_bias = 'True')
#			   (SCR_V6.4-13): Update HOModel initialization
#			   (SCR_V6.4-24): Skip convolution if the weights are all zero
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.4 (SCR_V6.3-2)
# Author/Date: Junseok Oh / 2019-03-29
# Change:      Make use of the Lookup table for the activation function Relu
# Cause:       Boost up the execution time of activation function Relu
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.4 (SCR_V6.3-1)
# Author/Date: Junseok Oh / 2019-03-24
# Change:      Make use of the Lookup table for the activation function STanh
# Cause:       Boost up the execution time of activation function STanh
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.3 (SCR_V6.2-1)
# Author/Date: Junseok Oh / 2019-03-17
# Change:      Sift out the values over the snLength
# Cause:       Boost up the execution time of add operation using MUX
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.2 (SCR_V6.1-4)
# Author/Date: Junseok Oh / 2019-02-19
# Change:      Define STanh and BTanh
# Cause:       Rename the activation functions
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.1 (SCR_V6.0-3)
# Author/Date: Junseok Oh / 2019-01-31
# Change:      Let convolution layer to count biases in the case of the normal model
# Cause:       Bug that convolution layer didn't count the bias value should be fixed
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.1 (SCR_V6.0-2)
# Author/Date: Junseok Oh / 2019-01-31
# Change:      Let convolution layer to ignore biases in the case of non-bias model
# Cause:       Handle non-bias model
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.1 (SCR_V6.0-1)
# Author/Date: Junseok Oh / 2019-01-31
# Change:      Delete the APC function which was replaced by SumUpAPC at V6.0
# Cause:       Unused function exists
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.0 (SCR_V5.4-2)
# Author/Date: Junseok Oh / 2019-01-10
# Change:      Perform the dense layer operations using binary number
# Cause:       For debugging purpose
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, //test.py
# Version:     6.0 (SCR_V5.4-1)
# Author/Date: Junseok Oh / 2019-01-01
# Change:      Create new methods in HOActivation
#              Let a user to select Mux-based or APC-based inner product
# Cause:       APC should be used for Conv and Dense layer
# Initiator:   Florian Neugebauer
################################################################################
# File:		   holayer.py, test.py
# Version:     5.4 (SCR_V5.3-3)
# Author/Date: Junseok Oh / 2018-11-20
# Change:      Change the order of weights in the convolution operation 
# Cause:       The order of weights in Keras is different with SNN
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
# Cause:       Implementing Object-Oriented Exact Stochastic MaxPooling
# Initiator:   Florian Neugebauer
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
# Change:      Change the conditions of the iteration in Activation  
# Cause:       Bug that it couldn't fill out the some areas of outputMatrix 
# Initiator:   Junseok Oh
###############################################################################
# Version:     1.0
# Author/Date: Junseok Oh / 2018-06-28
# Change:      Initial version
# Cause:       Encapsulate the activations
# Initiator:   Florian Neugebauer
###############################################################################

import numpy as np
from functools import reduce
import operator
import copy
import pickle

class HOLayer(object):
    def __call__(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, **kwargs):
        output = self.Call(inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, **kwargs)
        return output

    def Call(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, **kwargs):
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
        self.weightsSN = [ [] for t in range(1)]
        self.biasSN = [ [] for t in range(1)]
        self.denseWeightsSN = [0]
        self.denseBiasSN = [0]
        self.listIndex = [ [] for t in range(1)]

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

    def SetZeroBias(self, slices):
        # Slices refer to the number of filter
        self.biasSN = np.full((slices, 1), False)

    def SetDenseWeights(self, x):
        self.denseWeightsSN = x

    def SetDenseBias(self, x):
        self.denseBiasSN = x

    def SetZeroDenseBias(self, numClasses):
        self.denseBiasSN = np.zeros((1, numClasses))

    def SetListIndex(self, x):
        self.listIndex = x

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
                #tt = self.outputMatrix.size
                #tt1 = self.outputMatrix[0].size
                #tt2 = self.outputMatrix[0][0].size
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
                                                       self.weightsSN[i], self.biasSN[i], self.listIndex[i],
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
    def __init__(self, **kwargs):
        # Select the activation function to use
        for key in kwargs:
            if (key == "activationFunc"):
                self.activationFunc = kwargs[key]
            if (key == "use_bias"):
                self.use_bias = kwargs[key]

        # Generate the lookup table for 8bit, 16bit, and 25bit APC
        self.snLookupTableNumAPC8 = 0
        self.snLookupTableNumAPC16 = 0
        self.snLookupTableNumAPC25 = 0

        with open('snLookupTableNumAPC.pkl', 'rb') as input:
            self.snLookupTableNumAPC8 = pickle.load(input)
            self.snLookupTableNumAPC16 = pickle.load(input)
            _ = pickle.load(input)
            self.snLookupTableNumAPC25 = pickle.load(input)

        if (self.activationFunc == "Relu"):
            # Generate the lookup table for Relu activation function
            # 8bit [11111111] is equal to 255 in decimal
            self.snLookupTableNum = np.array(
                [self.GenerateLookupTableForRelu(byte) for byte in range(256)])

        elif(self.activationFunc == "STanh"):
            # Generate the lookup table for STanh activation function
            # the number of scale factors is 20. i.e. tanh(1*x), tanh(2*x), ... , tanh(20*x)
            # the number of states of STanh's state machine is determined by (num+1)*2
            # 8bit [11111111] is equal to 255 in decimal
            numScaleFactor = 20
            self.snLookupTableOut = [ []for num in range(numScaleFactor)]
            self.snLookupTableState = [ []for num in range(numScaleFactor)]
            for num in range(numScaleFactor):
                self.snLookupTableElementsTemp = np.array(
                    #[[self.GenerateLookupTableForSTanh(byte, state) for byte in range(256)] for state in range(32)])
                    [[self.GenerateLookupTableForSTanh(byte, state, (num+1)*2) for byte in range(256)] for state in range((num+1)*2)])
                self.snLookupTableOut[num] = copy.deepcopy(self.snLookupTableElementsTemp[:, :, 0])
                self.snLookupTableState[num] = copy.deepcopy(self.snLookupTableElementsTemp[:, :, 1])
            del(self.snLookupTableElementsTemp)

    def ActivationFuncReluLUTSN(self, x):
        # Represent the binary value into the 1byte decimal value
        sn = np.packbits(x)

        # Initialize the sum
        sum = 0

        for i, byte in enumerate(sn):
            # Find the element with the byte information in the table
            sum += self.snLookupTableNum[byte]

        if sum > len(x) / 2:
            return x
        else:
            return self.zeroSN

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

    def GenerateLookupTableForRelu(self, byte):
        # Represent the decimal value into 8bit binary value
        x = np.unpackbits(np.array([byte], dtype='uint8'))

        # Initialize the number of one
        numOne = 0
		
		# Count up the number of one if the bit is equal to 1
        for j, bit in enumerate(x):
            if(bit == 1):
                numOne = numOne +1

        return numOne

    def GenerateLookupTableForSTanh(self, byte, start_state, PAR_numState):
        # Set the number of states
        #numState = 32
        numState = PAR_numState

        # Represent the decimal value into 8bit binary value
        x = np.unpackbits(np.array([byte], dtype='uint8'))

        # Set state to start_state
        state = start_state

        for j, bit in enumerate(x):
            # Determine the output depending on the current state
            x[j] = state > (numState/2 -1)

            # input is True -> increase state
            if state < (numState-1) and bit:
                state += 1
            # input is False -> decrease state
            if state > 0 and not bit:
                state -= 1

        return (np.packbits(x)[0], state)

    def ActivationFuncSTanhLUTSN(self, x, PAR_numState):
        # Represent the binary value into the 1byte decimal value
        sn = np.packbits(x)
        out = np.empty_like(sn)

        # Set the number of states
        numState = PAR_numState * 2
        state = max(int(numState/2) - 1, 0)

        _snLookupTableOut = self.snLookupTableOut[PAR_numState-1]
        _snLookupTableState = self.snLookupTableState[PAR_numState-1]

        for i, byte in enumerate(sn):
            # Find the element with the current state and byte information in the table
            # Fill in the out value with that element
            out[i] = _snLookupTableOut[state, byte]
            # Update the state according to the element in the table
            state = _snLookupTableState[state, byte]

        # Represent the decimal value into 8bit binary value
        out = np.unpackbits(out)
        return out

    def ActivationFuncTanhSN(self, x, PAR_numState):
        # the number of states
        #numState = 32
        numState = PAR_numState*2
        # starting state
        state = max(int(numState/2)-1, 0)
        out = np.full(len(x), True, dtype=bool)
        for j in range(len(x)):
            # Determine the output depending on the current state
            if ((0 <= state) & (state < int(numState/2))):
                out[j] = 0
            else:
                out[j] = 1
            # input is True -> increase state
            if x[j] & (state < numState-1):
                state = state + 1
            # input is False -> decrease state
            elif np.logical_not(x[j]) & (state > 0):
                state = state - 1
            # No update at the start or end of the state
        #print("after stanh")
        #print(out)

        return out

    def SumUpAPCLUT(self, x):
        # Save the input in the buffer
        t1 = copy.deepcopy(x)
        t2 = copy.deepcopy(x)

        # The shape of the input x: (sizeTensor+sizeBias, snLength)
        size, snLength = x.shape

        # Find the required number of APCs
        numAPC8 = 0
        numAPC16 = 0
        numAPC25 = int(size / 25)

        # Check whether the pre-processing is needed or not
        sizePreprocessed = 0
        bPreprocessAPC8 = False
        bPreprocessAPC16 = False
        bPreprocessAPC25 = False
        if (0 < (size % 25) < 8):
            bPreprocessAPC8 = True
            numAPC8 = 1
        elif ((size % 25) == 8):
            numAPC8 = 1
        elif (8 < (size % 25) and (size % 25) < 16):
            bPreprocessAPC16 = True
            numAPC16 = 1
        elif ((size % 25) == 16):
            numAPC16 = 1
        elif (16 < (size % 25) and (size % 25) < 25):
            bPreprocessAPC25 = True
            numAPC25 += 1

        # Initialize the variable
        sum25 = np.full(snLength, 0)
        sum16 = np.full(snLength, 0)
        sum8 = np.full(snLength, 0)

        if (numAPC25 != 0):
            # Pre-process for the case where the size of input is less than 25bits
            if (bPreprocessAPC25 != False):
                sizePreprocessed = (25 * numAPC25) - size
                snZeros = [[] for i in range(sizePreprocessed)]
                for i in range(sizePreprocessed):
                    snZeros[i] = self.CreateSN(0, snLength)
                x = np.vstack((x, snZeros))

            # Remove the parts which are out of 25bit range
            x = x[:25 * numAPC25, :]

            # Transpose the input x: (snLength, sizeTensor+sizeBias)
            x = x.transpose()

            # Insert 7bit-zeros at every 25bits
            for i in range(numAPC25):
                for j in range(7):
                    x = np.insert(x, i * 32, False, axis=1)

            # Reshape it in order to pack in 16bits (4 x 8bits)
            x = x.reshape(snLength, -1, 4, 8)[:, :, ::-1]

            # Save the dimension information
            _, b, _, _ = x.shape

            # Pack the bits
            x = np.packbits(x).view(np.uint32)

            # Reshape it in order to handle the multiple APCs
            x = x.reshape(b, -1, order='F')

            # Look up the table
            for j in range(snLength):
                # Set the count number as 0
                jthSum = 0
                for i in range(numAPC25):
                    jthSum += self.snLookupTableNumAPC25[x[i, j]]
                sum25[j] = jthSum

        if (numAPC16 != 0):
            # Pre-process for the case where the size of input is less than 16bits
            if (bPreprocessAPC16 != False):
                sizePreprocessed = (25 * numAPC25) - size + 16
                snZeros = [[] for i in range(sizePreprocessed)]
                for i in range(sizePreprocessed):
                    snZeros[i] = self.CreateSN(0, snLength)
                t1 = np.vstack((t1, snZeros))

            t1 = t1[25 * numAPC25:(25 * numAPC25 + 16 * numAPC16), :]
            t1 = t1.transpose()
            t1 = t1.reshape(snLength, -1, 2, 8)
            _, b, _, _ = t1.shape
            t1 = np.packbits(t1).view(np.uint16)
            t1 = t1.reshape(b, -1, order='F')
            for j in range(snLength):
                jthSum = 0
                for i in range(numAPC16):
                    jthSum += self.snLookupTableNumAPC16[t1[i, j]]
                sum16[j] = jthSum

        if (numAPC8 != 0):
            # Pre-process for the case where the size of input is less than 8bits
            if (bPreprocessAPC8 != False):
                sizePreprocessed = (25 * numAPC25) - size + 8
                snZeros = [[] for i in range(sizePreprocessed)]
                for i in range(sizePreprocessed):
                    snZeros[i] = self.CreateSN(0, snLength)
                t2 = np.vstack((t2, snZeros))

            t2 = t2[(25 * numAPC25 + 16 * numAPC16):(25 * numAPC25 + 16 * numAPC16 + 8 * numAPC8), :]
            t2 = t2.transpose()
            t2 = t2.reshape(snLength, -1, 1, 8)
            _, b, _, _ = t2.shape
            t2 = np.packbits(t2).view(np.uint8)
            t2 = t2.reshape(b, -1, order='F')
            for j in range(snLength):
                jthSum = 0
                for i in range(numAPC8):
                    jthSum += self.snLookupTableNumAPC8[t2[i, j]]
                sum8[j] = jthSum

        sum = sum25 + sum16 + sum8

        return sum, sizePreprocessed, numAPC25, numAPC16, numAPC8


    def SumUpAPC(self, x, snLength, numAPC):
        # sizeState = r
        # snLength = m
        sum = np.full(snLength, 0)

        for j in range(snLength):
            # Set the count number as 0
            jthSum = 0

            # Count the number of 1s on each column approximately
            # and save the result in jthSum
            for i in range(numAPC):
                # AND, OR gates
                a = (x[0 + 16 * i, j] | x[1 + 16 * i, j])
                b = (x[2 + 16 * i, j] & x[3 + 16 * i, j])
                c = (x[4 + 16 * i, j] | x[5 + 16 * i, j])
                d = (x[6 + 16 * i, j] & x[7 + 16 * i, j])
                e = (x[8 + 16 * i, j] | x[9 + 16 * i, j])
                f = (x[10 + 16 * i, j] & x[11 + 16 * i, j])
                z2 = (x[12 + 16 * i, j] | x[13 + 16 * i, j])
                t0 = (x[14 + 16 * i, j] & x[15 + 16 * i, j])

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
                jthSum = jthSum + 8 * t3 + 4 * t2 + 2 * t1 + 2 * t0

            sum[j] = jthSum

        return sum

    def Count2Integer(self, x, snLength, numAPC25, numAPC16, numAPC8):
        sumTotal = 0

        for i in range(len(x)):
            sumTotal = sumTotal + x[i]

        ret = (sumTotal / snLength) * 2 - (25 * numAPC25) - (16 * numAPC16) - (8 * numAPC8)
        return ret

    def UpDownCounter(self, x, sizeTensor, sizeState):
        # the parameter sizeTensor here refers to (sizeTensor+sizeBias)

        # len(x) = m
        # sizeTensor = n
        # sizeState = r
        stateMax = sizeState - 1
        stateHalf = int(sizeState / 2)
        stateCurrent = stateHalf
        y = np.full(len(x), False)

        for i in range(len(x)):
            # Bipolar 1s counting
            v = x[i] * 2 - sizeTensor

            # Update the current state
            stateCurrent = stateCurrent + v

            # Check the exceptional cases
            if (stateCurrent > stateMax):
                stateCurrent = stateMax
            if (stateCurrent < 0):
                stateCurrent = 0

            # Generate the output using the stochastic number
            if (stateCurrent > stateHalf):
                y[i] = 1
            else:
                y[i] = 0

        return y

    def UpDownCounterReLU(self, x, sizeTensor, sizeState):
        # the parameter sizeTensor here refers to (sizeTensor+sizeBias)

        # len(x) = m
        # sizeTensor = n
        # sizeState = r
        stateMax = sizeState - 1
        stateHalf = int(sizeState / 2)
        stateCurrent = stateHalf
        y = np.full(len(x), False)
        accumulated = 0

        for i in range(len(x)):
            # Bipolar 1s counting
            v = x[i] * 2 - sizeTensor

            # Update the current state
            stateCurrent = stateCurrent + v

            # Check the exceptional cases
            if (stateCurrent > stateMax):
                stateCurrent = stateMax
            if (stateCurrent < 0):
                stateCurrent = 0

            # Enforce the output of ReLU to be greater than or equal to 0
            if (accumulated < int(i / 2)):
                y[i] = 1

            # Otherwise, the output is  determined by the following FSM
            else:
                # Generate the output using the stochastic number
                if (stateCurrent > stateHalf):
                    y[i] = 1
                else:
                    y[i] = 0

            # Accumulate the current output
            accumulated += y[i]

        return y


    def CreateSN(self, x, length):
        """create bipolar SN by comparing random vector elementwise to SN value x"""
        # rand = np.random.rand(length)*2.0 - 1.0
        # x_SN = np.less(rand, x)
        large = np.random.rand(1)
        x_SN = np.full(length, False)
        if large:
            for i in range(int(np.ceil(((x + 1) / 2) * length))):
                try:
                    x_SN[i] = True
                except IndexError:
                    print("The number is out of range (-1, +1)")
                    print("x: " + str(x))
        else:
            for i in range(int(np.floor(((x + 1) / 2) * length))):
                try:
                    x_SN[i] = True
                except IndexError:
                    print("The number is out of range (-1, +1)")
                    print("x: " + str(x))
        np.random.shuffle(x_SN)
        return x_SN

class HOMaxPooling(HOLayer):
    def Call(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, **kwargs):
        output = self.PoolingFunc(inputs)
        return output

    def PoolingFunc(self, inputs):
        raise NotImplementedError

class HOConv(HOActivation):
    def Call(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, **kwargs):
        output = self.ConvFunc(inputs, weights, bias, listIndex)
        return output

    def ConvFunc(self, inputs, weights, bias, listIndex):
        raise NotImplementedError

class HOConn(HOActivation):
    def Call(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, **kwargs):
        output = self.DenseFunc(inputs, numClasses, denseWeights, denseBias)
        return output

    def DenseFunc(self, inputs, numClasses, denseWeights, denseBias):
        raise NotImplementedError

class HOConnected(HOConn):
    def __init__(self, PAR_SnLength, **kwargs):
        super().__init__(**kwargs)

        self.SetLayerID("FullyConnected")
        self.snLength = PAR_SnLength
        self.dense_output_SN = [0]

        # Select the way of transforming stochastic number to integer
        for key in kwargs:
            if (key == "stochToInt"):
                self.stochToInt = kwargs[key]

    def DenseFunc(self, inputs, numClasses, denseWeights, denseBias):
        numInputPlanes, sizeRow, sizeCol, snLength = inputs.shape

        # Flatten the inputs
        sizeTensor = numInputPlanes * sizeRow * sizeCol  # The total number of elements in the whole layer
        #sizeTensor = int (inputs.size / inputs[0][0][0].size) # The total number of elements in the whole layer
        denseInputs = inputs.reshape((1, sizeTensor, self.snLength))

        self.dense_output_SN = np.full((numClasses, sizeTensor, self.snLength), False)
        self.dense_output = np.zeros((1, numClasses))
        ################### For debugging purpose,  Create temporal variables ###################
        #self.dense_output_BN = np.zeros((numClasses, sizeTensor))
        #self.dense_output_binary = np.zeros((1, numClasses))
        ################################################################################################################		

        # PRODUCT in the inner product operations
        for i in range(sizeTensor):
            for j in range(numClasses):
                self.dense_output_SN[j, i] = np.logical_not(np.logical_xor(denseInputs[0, i], denseWeights[i, j]))
        ################### For debugging purpose,  Perform PRODUCT operation using binary number ###################
        #        self.dense_output_BN[j, i] = self.StochToInt(denseInputs[0, i]) * self.StochToInt(denseWeights[i, j])
        ################################################################################################################

        # ADD in the inner product operations
        ################### For debugging purpose,  Add all results of PRODUCTs using binary number ###################
        #for i in range(numClasses):
        #    for j in range(sizeTensor):
        #        self.dense_output_binary[0, i] = self.dense_output_binary[0, i] + self.dense_output_BN[i, j]
        ################################################################################################################

        if (self.stochToInt == "Normal"):
            for i in range(numClasses):
                for j in range(sizeTensor):
                    self.dense_output[0, i] = self.dense_output[0, i] + self.StochToInt(self.dense_output_SN[i, j])

        elif (self.stochToInt == "APC"):
            count = np.full(self.snLength, 0)
            for i in range(numClasses):
                #self.dense_output[0, i] = self.APC(self.dense_output_SN[i], self.snLength, sizeTensor)
                # count = self.SumUpAPC(self.dense_output_SN[i], self.snLength, numAPC)
                count, _, numAPC25, numAPC16, numAPC8  = self.SumUpAPCLUT(self.dense_output_SN[i])
                self.dense_output[0, i] = self.Count2Integer(count, self.snLength, numAPC25, numAPC16, numAPC8)
                del(count)

        # Biasing
        self.dense_output = self.dense_output + denseBias
        ################### For debugging purpose,  Add biases using binary number ###################
        #self.dense_output_binary = self.dense_output_binary + denseBias
        #print("For debugging purpose, dense_output using binary number are as follows")
        #print(self.dense_output_binary)
		################################################################################################################		

        # Activation function
        if (self.activationFunc == "Relu"):
            for i in range(len(self.dense_output[0])):
                self.dense_output[0][i] = self.ActivationFuncRelu(self.dense_output[0][i])

        elif (self.activationFunc == "Tanh"):
            '''Not yet implemented'''
            #self.dense_output = self.ActiavtionFuncTanh(self.dense_output)
            pass
        elif (self.activationFunc == "None"):
            pass

        return self.dense_output

    def StochToInt(self, x):
        """convert bipolar stochastic number to integer"""
        return (sum(x) / len(x)) * 2.0 - 1.0

class HOConvolution(HOConv):
    def __init__(self, PAR_Row, PAR_Col, PAR_SnLength, **kwargs):
        super().__init__(**kwargs)

        # Select the base mode, MUX or APC
        for key in kwargs:
            if (key == "baseMode"):
                self.baseMode = kwargs[key]

        # parameter setting
        self.SetLayerID("Convolution")
        self.row = PAR_Row
        self.col = PAR_Col
        self.matrixSize = PAR_Row * PAR_Col
        self.snLength = PAR_SnLength


        # Initialize the Convolution output
        self.listProductSN = 0
        self.convOutput = np.full((1, self.snLength), False)
        #self.convOutputDebug = np.zeros((1))

        # Create the Stochastic Number Zero
        self.zeroSN = self.CreateSN(0, self.snLength)

    def GetFilterSize(self):
        return self.row

    def StochToInt(self, x):
        """convert bipolar stochastic number to integer"""
        return (sum(x) / len(x)) * 2.0 - 1.0

    def ConvFunc(self, inputs, weights, bias, listIndex):
        numInputPlanes, sizeRow, sizeCol, snLength = inputs.shape

        # Determine whether the layer uses a bias vector
        if(self.use_bias == 'True'):
            sizeBias = 1
        else:
            sizeBias = 0

        # Determine the size of tensors
        sizeTensor = sizeBias + (numInputPlanes * sizeRow * sizeCol)
        # numInputPlanes = int (inputs.size / inputs[0].size) # The number of planes in the inputs
        # sizeTensor = int(inputs.size / inputs[0][0][0].size) # it is equal to (numInputPlanes * self.matrixSize)

        # Initialize the Convolution output
        self.listProductSN = np.full((sizeTensor, self.snLength), False)

        # PRODUCT in the inner product operations
        for k in range(numInputPlanes):
            for i in range(self.row):
                for j in range(self.col):
                    self.listProductSN[k*self.matrixSize+i*self.col+j] = np.logical_not(np.logical_xor(weights[k, i, j],
                                                                                                       inputs[k, self.row-1-i, self.col-1-j]))
        # Put the bias in the last element of listProductSN if the bias exists
        if(self.use_bias == 'True'):
            self.listProductSN[sizeTensor-1] = bias

        # Make use of listIndex not to consider zero weights in addition operation
        sizeTensorCompact = len(listIndex)
        self.listProductSNCompact = np.full((sizeTensorCompact, self.snLength), False)
        for i in range(sizeTensorCompact):
            self.listProductSNCompact[i] = self.listProductSN[listIndex[i]]

        # ADD in the inner product operations
        if (self.baseMode == "Mux"):
            s = np.full((sizeTensorCompact, self.snLength),False)
            self.convOutput[0] = np.full((self.snLength), False)

            # Do not skip convolution if an one of the weights is not zero
            if(sizeTensorCompact != 0):
                # Generate random numbers that determine which input will be selected in the mux
                r = np.random.randint(0, sizeTensorCompact, self.snLength)

                for i in range(sizeTensorCompact):
                    # Make a filter from the random number set
                    s[i] = (r == i).astype(int)
                    # Sift out the values in the listProductSN over the snLength
                    self.convOutput[0] |= self.listProductSNCompact[i] & s[i]
            # Skip Convolution if the weights are all zero
            else:
                self.convOutput[0] = self.zeroSN

        elif(self.baseMode == "APC"):
            count = np.full(self.snLength, 0)
            #numAPC = int((sizeTensor + sizeBias) / 16)
            #count = self.SumUpAPC(self.listProductSN, self.snLength, numAPC)
            count, sizePreprocessed, _, _, _ = self.SumUpAPCLUT(self.listProductSN)
            # Debugging purpose#######################################################
            #if (numInputPlanes != 1):
            #    self.convOutputDebug = self.Count2Integer(count, self.snLength, numAPC)
            #    print(self.convOutputDebug)
            #    self.convOutput[0] = self.CreateSN(self.convOutputDebug, self.snLength)
            ##########################################################################

        # Activation function
        if(self.activationFunc == "Relu"):
            #print("start Relu")
            self.convOutput[0] = self.ActivationFuncReluLUTSN(self.convOutput[0])
            #self.convOutput[0] = self.ActivationFuncReluSN(self.convOutput[0])
            #print("end Relu")
        elif(self.activationFunc == "SCRelu"):
            self.convOutput[0] = self.UpDownCounterReLU(count, (sizeTensor+sizePreprocessed), 4*(sizeTensor+sizePreprocessed))
            # Debugging purpose#######################################################
            # numAPC25 = int(sizeTensor / 25)
            # numAPC16 = int((sizeTensor % 25) / 16)
            # numAPC8 = int(((sizeTensor % 25) % 16) / 8)
            # tempInteger = np.zeros(1)
            # tempInteger[0] = self.Count2Integer(count, self.snLength, numAPC25, numAPC16, numAPC8)
            # if(tempInteger[0] > 1):
            #     tempInteger[0] = 1
            # elif(tempInteger[0] < -1):
            #     tempInteger[0] = -1
            # self.convOutput[0] = self.CreateSN(tempInteger[0], self.snLength)
            ##########################################################################

        elif(self.activationFunc == "STanh"):
            #print("start STanh")
            if(sizeTensorCompact != 0):
                self.convOutput[0] = self.ActivationFuncSTanhLUTSN(self.convOutput[0], sizeTensorCompact)
                #self.convOutput[0] = self.ActivationFuncTanhSN(self.convOutput[0], sizeTensorCompact)
            #print("end STanh")
        elif(self.activationFunc == "BTanh"):
            # Debugging purpose#######################################################
            #if (numInputPlanes == 1):
            #    self.convOutput[0] = self.UpDownCounter(count, (sizeTensor + sizeBias),2 * (sizeTensor + sizeBias))  # 1/s = 1
            #else:
            #    pass
            ##########################################################################
            self.convOutput[0] = self.UpDownCounter(count, (sizeTensor+sizePreprocessed), 2*(sizeTensor+sizePreprocessed)) # 1/s = 1

        return self.convOutput


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
