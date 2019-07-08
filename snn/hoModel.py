#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     hoModel.py
#
###############################################################################
#  Description:
#  
#  (For a detailed description look at the object description in the UML model)
#  
###############################################################################
# History
################################################################################
# File:		   hoModel.py
# Version:     15.0
# Author/Date: Junseok Oh / 2019-07-01
# Change:      (SCR_V14.0-1): Modularize the classes, change the file names
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################

import numpy as np
from snn.hoSnn import HOSnn

class HOModel(HOSnn):
    def __init__(self, inputMatrix, **kwargs):
        super().__init__(**kwargs)

        # Calibration values
        self.padding = 0

        # parameter setting
        self.numInputPlanes = 1
        self.numOutputPlanes = 1
        self.inputWidth = int(inputMatrix.size / inputMatrix[0].size)

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
                self.numInputClasses = int(self.outputMatrix[0].size / self.outputMatrix[0][0].size)
                resMatrix = np.full((self.numInputClasses, self.snLength), False)
                for i in range(self.numInputClasses):
                    resMatrix[i] = self.CreateSN(self.outputMatrix[0, i])
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