###############################################################################
#                                                                             #
#                            Copyright (c)                                    #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:	hoLayer.py
#  Description:	
#  Author/Date:	Junseok Oh / 2020-02-27
#  Initiator:	Florian Neugebauer
################################################################################
	
import numpy as np
import copy
import pickle
from snn.hoSnn import HOSnn

"""
Architecture of the classes
  HOSnn
	|			
 HOLayer	
	|
	------------------------------------------------------
	|				  									 |
HOMaxPooling											HOActivation
	|													 |
	-------------------------------------				 ----------------
	|									|				 |			    |
HOMaxPoolingAprox				HOMaxPoolingExact	  HOConv		 HOConn
	|													 |      	  	|
------------------------							HOConvolution	HOConnected
1		   1		   *
Mux	  Comparator	Counter			
"""


class HOLayer(HOSnn):
    def __call__(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, listIndexDense, **kwargs):
        output = self.Call(inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, listIndexDense, **kwargs)
        return output

    def Call(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, listIndexDense, **kwargs):
        """
        the instance of HOModel calls the instance of HOLayer
        The specific implementations are specified according to the type of HOLayer instance as follows
        HOConnected: 		Call()-> DenseFunc()
        HOConvolution: 		Call()-> ConvFunc()
        HOMaxPoolingExact: 	Call()-> PoolingFunc()
        HOMaxPoolingAprox: 	Call()-> PoolingFunc()
        """
        #This is where the layer's logic lives.
        return inputs

    def SetLayerID(self, x):
        self.layerID = x

    def GetLayerID(self):
        return self.layerID


class HOActivation(HOLayer):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        modeAddConv: string
            the type of how to implement addition operations in convolution layers with stochastic circuits
            Possible elements: "Mux", "APC"

        modeAddConn: string
            the type of how to implement addition operations in connected layers with stochastic circuits
            Possible elements: "Normal", "Mux", "APC"
            With "Normal" mode, SNs are converted to BNs to calculate additions

        activationFunc: string
            the type of the activation function of a convolution or connected layer
            Possible elements: "None", "Relu", "ReluByMax", "SCRelu", "STanh", "BTanh"
            With "Relu", the number of zeros in SN is exhaustly counted
            With "ReluByMax", Exact max-pooling stochastic circuit is used to perform Relu
            With "SCRelu", Up/down counter is used to perform Relu after additions implemented by "APC"
            With "STanh", K-state FSM is used to perform Tanh after additions implemented by "Mux"
            With "BTanh", Up/down counter is used to perform Tanh after additions implemented by "APC"

        use_bias: string
            the usage of bias values at a convolution or connected layer
            Possible elements: "True", "False"

        scale: float
            the scaling factor of the activation function

        constantH: float
            the value of constant H for "BTanh"

        Examples of convolution layer configurations
        -------------------------------------------------
        SC for addition operations | Activation function
        (modeAddConv)              | (activationFunc)
        -------------------------------------------------
        "Mux"					   | "Relu"
        "Mux"                      | "ReluByMax"
        "Mux"					   | "STanh"
        "APC"					   | "SCRelu"
        "APC"					   | "BTanh"

        Examples of connected layer configurations
        -------------------------------------------------
        SC for addition operations | Activation function
        (modeAddConn)              | (activationFunc)
        -------------------------------------------------
        "Normal"				   | "None"
        "Normal"				   | "Relu"
        "Mux"					   | "None"
        "Mux"					   | "Relu"
        "APC"					   | "None"
        "APC"					   | "Relu"
        """
        super().__init__(**kwargs)

        # Set scale factor 1 as default
        self.scale = 1

        # Set constantH 0.8 as default
        self.constantH = 0.8

        # Select the activation function to use
        for key in kwargs:
            if (key == "modeAddConv"):
                self.modeAddConv = kwargs[key]
            if (key == "modeAddConn"):
                self.modeAddConn = kwargs[key]
            if (key == "activationFunc"):
                self.activationFunc = kwargs[key]
            if (key == "use_bias"):
                self.use_bias = kwargs[key]
            if (key == "scale"):
                self.scale = kwargs[key]
            if (key == "constantH"):
                self.constantH = kwargs[key]

        # Generate the lookup table for 8bit, 16bit, and 25bit APC
        self.snLookupTableNumAPC8 = 0
        self.snLookupTableNumAPC16 = 0
        self.snLookupTableNumAPC25 = 0

        try:
            if (self.modeAddConv == "APC"):
                with open('../snLookupTableNumAPC.pkl', 'rb') as input:
                    self.snLookupTableNumAPC8 = pickle.load(input)
                    self.snLookupTableNumAPC16 = pickle.load(input)
                    _ = pickle.load(input)
                    self.snLookupTableNumAPC25 = pickle.load(input)
        except AttributeError:
            pass

        try:
            if (self.modeAddConn == "APC"):
                with open('../snLookupTableNumAPC.pkl', 'rb') as input:
                    self.snLookupTableNumAPC8 = pickle.load(input)
                    self.snLookupTableNumAPC16 = pickle.load(input)
                    _ = pickle.load(input)
                    self.snLookupTableNumAPC25 = pickle.load(input)
        except AttributeError:
            pass

        try:
            if (self.activationFunc == "Relu"):
                # Generate the lookup table for Relu activation function
                # 8bit [11111111] is equal to 255 in decimal
                with open('../snLookupTableNumRelu.pkl', 'rb') as input:
                    self.snLookupTableNum = pickle.load(input)

            elif(self.activationFunc == "STanh"):
                # Generate the lookup table for STanh activation function
                # the number of scale factors is 20. i.e. tanh(1*x), tanh(2*x), ... , tanh(20*x)
                # the number of states of STanh's state machine is determined by (num+1)*2
                # 8bit [11111111] is equal to 255 in decimal
                with open('../snLookupTableNumTanh.pkl', 'rb') as input:
                    self.snLookupTableOut = pickle.load(input)
                    self.snLookupTableState = pickle.load(input)
        except AttributeError:
            pass

    def ActivationFuncReluLUTSN(self, x):
        """
        The activation function ReLU implemented by Look-Up Table and stochastic numbers
        """
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

    def ActivationFuncRelu(selfs, x):
        """
        The activation function ReLU implemented by binary numbers
        """
        if (x <= 0):
            return 0
        else:
            return x

    def ActivationFuncSTanhLUTSN(self, x, PAR_numState, PAR_scale):
        """
        The activation function Tanh implemented by K-state FSM, Look-Up table, and stochastic numbers
        """
        # Represent the binary value into the 1byte decimal value
        sn = np.packbits(x)
        out = np.empty_like(sn)

        # Set the number of states
        numState = PAR_numState * PAR_scale * 2
        state = max(int(numState/2) - 1, 0)

        _snLookupTableOut = self.snLookupTableOut[state]
        _snLookupTableState = self.snLookupTableState[state]

        for i, byte in enumerate(sn):
            # Find the element with the current state and byte information in the table
            # Fill in the out value with that element
            out[i] = _snLookupTableOut[state, byte]
            # Update the state according to the element in the table
            state = _snLookupTableState[state, byte]

        # Represent the decimal value into 8bit binary value
        out = np.unpackbits(out)
        return out

    def SumUpAPCLUT(self, x):
        """
        Addition operation implemented by APC8, APC16, and APC25 with Look-Up Table

        Returns
        -------
        sum: array
            the total column sum of APCs

        sizePreprocessed: int
            the number of SN zeros to completly fill up the inputs of each APC
            e.g. If the input size is 15 and one 16bit APC is supposed to be used,
                then sizePreprocessed is 1 (= 16-15)

        numAPC25: int
            the required number of 25bit APCs

        numAPC16: int
            the required number of 16bit APCs

        numAPC8: int
            the required number of 8bit APCs

        """
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
                    snZeros[i] = self.CreateSN(0)
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
                    snZeros[i] = self.CreateSN(0)
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
                    snZeros[i] = self.CreateSN(0)
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

    def Count2Integer(self, x, snLength, numAPC25, numAPC16, numAPC8):
        """
        Converting count of APCs to integer
        """
        sumTotal = 0

        for i in range(len(x)):
            sumTotal = sumTotal + x[i]

        ret = (sumTotal / snLength) * 2 - (25 * numAPC25) - (16 * numAPC16) - (8 * numAPC8)
        return ret

    def UpDownCounter(self, x, sizeTensor, constantH, scale):
        """
        Up/down counter for activation function "BTanh" after addition operations implemented by APCs
        """
        # the parameter sizeTensor here refers to (sizeTensor+sizeBias)

        # len(x) = m
        # sizeTensor = n

        # sizeState = r
        # scale = 1/s * constantH
        # 1/s = 1 + (r'-2n)*(1-1.835*(2n)^-0.5552)/2/(n-1)
        # r' = 2n + (-1+scale/constantH)*2*(n-1) / (1-1.835*(2n)^-0.5552)
        # r = nearest_multiple_of_two(r')
        sizeState = round((2*sizeTensor + (-1+scale/constantH)*2*(sizeTensor-1) / (1-1.835*(2*sizeTensor) ** -0.5552))/2)*2

        stateMax = sizeState - 1
        stateHalf = int(sizeState / 2) - 1
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

    def UpDownCounterReLU(self, x, sizeTensor, constantH=0.8, scale=1.232):
        """
        Up/down counter for the activation function "SCRelu" after addition operations implemented by APCs
        """
        # the parameter sizeTensor here refers to (sizeTensor+sizeBias)

        # len(x) = m
        # sizeTensor = n

        # sizeState = r
        # scale = 1/s * constantH
        # 1/s = 1 + (r'-2n)*(1-1.835*(2n)^-0.5552)/2/(n-1)
        # r' = 2n + (-1+scale/constantH)*2*(n-1) / (1-1.835*(2n)^-0.5552)
        # r = nearest_multiple_of_two(r')
        sizeState = round((2*sizeTensor + (-1+scale/constantH)*2*(sizeTensor-1) / (1-1.835*(2*sizeTensor) ** -0.5552))/2)*2

        stateMax = sizeState - 1
        stateHalf = int(sizeState / 2) - 1
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


class HOConn(HOActivation):
    def Call(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, listIndexDense, **kwargs):
        output = self.DenseFunc(inputs, numClasses, denseWeights, denseBias, listIndexDense)
        return output

    def DenseFunc(self, inputs, numClasses, denseWeights, denseBias, listIndexDense):
        raise NotImplementedError


class HOConnected(HOConn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.SetLayerID("FullyConnected")

        # Create the Stochastic Number Zero
        self.zeroSN = self.CreateSN(0)

    def DenseFunc(self, inputs, numClasses, denseWeights, denseBias, listIndexDense):
        """
        Performing convolution operations in a fully-connected layer
        It is supposed to be called by "Run" function of HOModel
        The interface of DenseFunc is "Call" function of HOLayer

        Parameters
        ----------
        inputs: object
            inputs of the fully-connected layer over which a kernel would perform its convolution operation

        numClasses: int
            the number of output classes of the fully-connected layer

        denseWeights: object
            the weights of the fully-connected layer

        denseBias: object
            the biases of the fully-connected layer

        listIndexDense: object (two-dimensional list)
            the list of indices which indicate the positions of non-near-zero weights
            e.g. [ [1, 3, 4], [2, 5] ]  indicates the followings:
                # the number of classes is two
                # 1st, 3rd, and 4th weight of the first class are not zero
                # 2nd and 5th weight of the second class are not zero

        Returns
        -------
        dense_output: object
            outputs of the fully-connected layer
        """
        numInputPlanes, sizeRow, sizeCol, snLength = inputs.shape

        # Determine whether the layer uses a bias vector
        if(self.use_bias == 'True'):
            sizeBias = 1
        else:
            sizeBias = 0

        # Flatten the inputs
        sizeTensor = sizeBias + (numInputPlanes * sizeRow * sizeCol)  # The total number of elements in the whole layer
        denseInputs = inputs.reshape((1, sizeTensor - sizeBias, self.snLength))

        self.dense_Product_SN = np.full((numClasses, sizeTensor, self.snLength), False)
        self.dense_output_SN = np.full((numClasses, snLength), False)
        self.dense_output = np.zeros((1, numClasses))

        # PRODUCT in the inner product operations
        for i in range(sizeTensor - sizeBias):
            for j in range(numClasses):
                self.dense_Product_SN[j, i] = np.logical_not(np.logical_xor(denseInputs[0, i], denseWeights[i, j]))

        # Put the biases in the last elements of dense_output_SN if the biases exist
        if(self.use_bias == 'True'):
            for j in range(numClasses):
                self.dense_Product_SN[j, sizeTensor - sizeBias] = denseBias[j]

        # ADD in the inner product operations
        if (self.modeAddConn == "Normal"):
            for i in range(numClasses):
                for j in range(sizeTensor):
                    self.dense_output[0, i] = self.dense_output[0, i] + self.StochToInt(self.dense_Product_SN[i, j])

        # ADD in the inner product operations
        elif (self.modeAddConn == "Mux"):
            # Make use of listIndex not to consider zero weights in addition operation
            for j in range(numClasses):
                sizeTensorCompact = len(listIndexDense[j])
                self.dense_Product_SNCompact = np.full((sizeTensorCompact, self.snLength), False)
                for i in range(sizeTensorCompact):
                    indexTemp = listIndexDense[j][i]
                    self.dense_Product_SNCompact[i] = self.dense_Product_SN[j, indexTemp]

                s = np.full((sizeTensorCompact, self.snLength), False)

                # Do not skip convolution if an one of the weights is not zero
                if(sizeTensorCompact != 0):

                    # Generate random numbers that determine which input will be selected in the mux
                    r = np.random.randint(0, sizeTensorCompact, self.snLength)

                    for i in range(sizeTensorCompact):
                        # Make a filter from the random number set
                        s[i] = (r == i).astype(int)
                        # Sift out the values in the listProductSN over the snLength
                        self.dense_output_SN[j] |= self.dense_Product_SNCompact[i] & s[i]

                # Skip Convolution if the weights are all zero
                else:
                    self.dense_output_SN[j] = self.zeroSN

                # convert dense_output_SN into dense_output
                self.dense_output[0, j] = self.StochToInt(self.dense_output_SN[j])

        elif (self.modeAddConn == "APC"):
            for i in range(numClasses):
                count, _, numAPC25, numAPC16, numAPC8  = self.SumUpAPCLUT(self.dense_Product_SN[i])
                self.dense_output[0, i] = self.Count2Integer(count, self.snLength, numAPC25, numAPC16, numAPC8)
                del(count)

        # Activation function
        if (self.activationFunc == "Relu"):
            for i in range(len(self.dense_output[0])):
                self.dense_output[0][i] = self.ActivationFuncRelu(self.dense_output[0][i])

        elif (self.activationFunc == "None"):
            pass

        return self.dense_output


class HOConv(HOActivation):
    def Call(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, listIndexDense, **kwargs):
        output = self.ConvFunc(inputs, weights, bias, listIndex)
        return output

    def ConvFunc(self, inputs, weights, bias, listIndex):
        raise NotImplementedError


class HOConvolution(HOConv):
    def __init__(self, PAR_Row, PAR_Col, **kwargs):
        super().__init__(**kwargs)

        # parameter setting
        self.SetLayerID("Convolution")
        self.row = PAR_Row
        self.col = PAR_Col
        self.matrixSize = PAR_Row * PAR_Col

        # Initialize the Convolution output
        self.listProductSN = 0
        self.convOutput = np.full((1, self.snLength), False)

        # Create the Stochastic Number Zero
        self.zeroSN = self.CreateSN(0)

        # Create an instance of HOMaxPoolingExact when ReluByMax is selected as the activation function
        if(self.activationFunc == "ReluByMax"):
            self.reluByMax = HOMaxPoolingExact(1, 2, kBits=self.kBits)
            self.inputsMax = np.full((1, 2, self.snLength), False)
            self.inputsMax[0, 0] = self.zeroSN

    def GetFilterSize(self):
        return self.row

    def ConvFunc(self, inputs, weights, bias, listIndex):
        """
        Performing convolution operations in a convolution layer
        It is supposed to be called by "Run" function of HOModel
        The interface of ConvFunc is "Call" function of HOLayer

        Parameters
        ----------
        inputs: object
            inputs of the convolution layer over which a kernel would perform its convolution operation

        weights: object
            the weights of the convolution layer

        bias: object
            the biases of the convolution layer

        listIndex: object (two-dimensional list)
            the list of indices which indicate the positions of non-near-zero weights
            e.g. [ [1, 3, 4], [2, 5] ]  indicates the followings:
                # the number of outputs is two
                # 1st, 3rd, and 4th weight of the first output are not zero
                # 2nd and 5th weight of the second output are not zero

        Returns
        -------
        convOutput: object
            outputs of the convolution layer
        """
        numInputPlanes, sizeRow, sizeCol, snLength = inputs.shape

        # Determine whether the layer uses a bias vector
        if(self.use_bias == 'True'):
            sizeBias = 1
        else:
            sizeBias = 0

        # Determine the size of tensors
        sizeTensor = sizeBias + (numInputPlanes * sizeRow * sizeCol)

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

        # ADD in the inner product operations
        if (self.modeAddConv == "Mux"):
            # Make use of listIndex not to consider zero weights in addition operation
            sizeTensorCompact = len(listIndex)
            self.listProductSNCompact = np.full((sizeTensorCompact, self.snLength), False)
            for i in range(sizeTensorCompact):
                self.listProductSNCompact[i] = self.listProductSN[listIndex[i]]

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

        elif(self.modeAddConv == "APC"):
            count, sizePreprocessed, _, _, _ = self.SumUpAPCLUT(self.listProductSN)

        # Activation function
        if(self.activationFunc == "Relu"):
            self.convOutput[0] = self.ActivationFuncReluLUTSN(self.convOutput[0])

        elif(self.activationFunc == "ReluByMax"):
            self.inputsMax[0, 1] = self.convOutput[0]
            resultMax = self.reluByMax.PoolingFunc(self.inputsMax)
            self.convOutput[0] = resultMax[0]

        elif(self.activationFunc == "SCRelu"):
            self.convOutput[0] = self.UpDownCounterReLU(count, (sizeTensor+sizePreprocessed))

        elif(self.activationFunc == "STanh"):
            if(sizeTensorCompact != 0):
                self.convOutput[0] = self.ActivationFuncSTanhLUTSN(self.convOutput[0], sizeTensorCompact, self.scale)

        elif(self.activationFunc == "BTanh"):
            self.convOutput[0] = self.UpDownCounter(count, (sizeTensor+sizePreprocessed), self.constantH, self.scale)

        return self.convOutput


class HOMaxPooling(HOLayer):
    def Call(self, inputs, weights, bias, listIndex, numClasses, denseWeights, denseBias, listIndexDense, **kwargs):
        output = self.PoolingFunc(inputs)
        return output

    def PoolingFunc(self, inputs):
        raise NotImplementedError


class HOMaxPoolingExact(HOMaxPooling):
    def __init__(self, PAR_Row, PAR_Col, **kwargs):
        super().__init__(**kwargs)

        # parameter setting
        self.SetLayerID("MaxPooling")
        self.row = PAR_Row
        self.col = PAR_Col
        self.matrixSize = PAR_Row * PAR_Col

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
        self.maxOutput = np.full((1, self.snLength), False)

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
        """
        Performing exact max-pooling operations
        It is supposed to be called by "Run" function of HOModel
        The interface of PoolingFunc is "Call" function of HOLayer

        Parameters
        ----------
        inputs: object
            inputs of the pooling layer

        Returns
        -------
        maxOutput: object
            outputs of the pooling layer
        """
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
    def __init__(self, PAR_Row, PAR_Col, PAR_step, **kwargs):
        super().__init__(**kwargs)

        # parameter setting
        self.SetLayerID("MaxPooling")
        self.row = PAR_Row
        self.col = PAR_Col
        self.matrixSize = PAR_Row * PAR_Col
        self.ithSN = np.random.rand(1) * (self.matrixSize-1)
        self.ithSN = int(np.ceil(self.ithSN[0]))
        self.numPartialSN = int (self.snLength / PAR_step)
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

        # Initialize the MaxPooling output
        self.maxOutput = np.full((1, self.snLength), False)

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
        """
        Performing approximate max-pooling operations
        It is supposed to be called by "Run" function of HOModel
        The interface of PoolingFunc is "Call" function of HOLayer

        Parameters
        ----------
        inputs: object
            inputs of the pooling layer

        Returns
        -------
        maxOutput: object
            outputs of the pooling layer
        """
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