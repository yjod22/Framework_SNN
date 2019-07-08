#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     bnLayer.py
#
###############################################################################
#  Description:
#  
#  (For a detailed description look at the object description in the UML model)
#  
###############################################################################
# History
################################################################################
# File:		   bnLayer.py
# Version:     15.0
# Author/Date: Junseok Oh / 2019-07-01
# Change:      (SCR_V14.0-1): Modularize the classes, change the file names
#              (SCR_V14.0-7): Allow user to set the number of possible iterations
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   bnlayer.py
# Version:     12.0 
# Author/Date: Junseok Oh / 2019-06-27
# Change:      (SCR_V11.0-7): Change the whole sw structure
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   bnlayer.py
# Version:     9.0
# Author/Date: Junseok Oh / 2019-06-07
# Change:      (SCR_v8.0-7): Fix bug in finding the number of conv layers
#              (SCR_V8.0-9): Fix bug in GetModel
# Cause:       bug fix
# Initiator:   Junseok Oh
################################################################################
# File:		   bnlayer.py
# Version:     8.0
# Author/Date: Junseok Oh / 2019-05-23
# Change:      (SCR_v6.4-18): Created
#              (SCR_V6.4-20): Fix bug in std deviation
#              (SCR_V6.4-21): Re-train the model again if it fails
#              (SCR_V6.4-22): Create PlotCurrentWeights
#              (SCR_V6.4-23): Upgrade in replacing
#              (SCR_V6.4-27): Upgrade in replacing
#              (SCR_V6.4-28): Encapsulate the functions
#              (SCR_V6.4-30): Upgrade in replacing (use intMax, limit the number of iteration, load 1st model back when retraining fails)
#              (SCR_V6.4-37): Upgrade in replacing (the number of non-zero elements must be smaller than before, sign change after permutation)
#              (SCR_V6.4-38): Upgrade in replacing (user-defined the number of iteration, handle multiple layers)
#              (SCR_V6.4-40): Upgrade in replacing (use intAvg to determine further iteration)
#              (SCR_V6.4-41): Bug fix, Save and Load weights in / from 1st Model
# Cause:       new
# Initiator:   Junseok Oh
###############################################################################
import keras
from keras.models import Sequential, Model
import numpy as np
import random
import copy
from snn.hoUtils import HOUtils
import global_variables
import statistics

class BNModel(object):
    def __init__(self, numLayer):
        self.id = 0
        self.model = Sequential()  # The model has its own weights repository in itself.
        self.numLayer = numLayer
        self.layer = [None] * numLayer
        self.weights = []  # This is the weights for replacing
        self.weightsShadow = [] # The weights of 1st model with low epochs are saved here
        self.weights1stModel = [] # The weights of 1st model with highest epochs are saved here
        self.numWeightsLayer = []
        self.numRow = []
        self.numCol = []
        self.numInputSlices = []
        self.numOutputSlices = []
        self.listAvg = []
        self.mean = []
        self.stdDeviation = []
        self.listIndexOutliers = []
        self.listIndexNonOutliers = []
        self.listIndexNonZeroElements= []
        self.cntPossibleIteration = 0
        self.cntReplaceOutliers = 0

    def __setitem__(self, key, value):
        self.layer[key] = value

    def __getitem__(self, key):
        key += 1
        self.layerModel = Model(inputs=self.model.input, outputs=self.model.get_layer(index=key).output)
        return copy.deepcopy(self.layerModel)

    def GetModel(self):
        #return copy.deepcopy(self.model)
        return self.model

    def GetId(self):
        return self.id

    def SetId(self, x):
        self.id = x

    def LoadLayers(self):
        # Remove all layers that have been already loaded
        for key in range(len(self.layer)):
            try:
                self.model.pop()
            except TypeError:
                pass
        # Load all layers onto the model
        for key in range(len(self.layer)):
            self.model.add(self.layer[key])

    def Compile(self, optimizer, loss, metrics, **kwargs):
        # Compile the model
        self.model.compile(optimizer,
                           loss,
                           metrics=metrics,
                           **kwargs)

    def Fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_data, **kwargs):
        self.model.fit(x=x,
                       y=y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks,
                       validation_data=validation_data,
                       **kwargs)

    def Evaluate(self, x, y, verbose, indexModel):
        self.score = self.model.evaluate(x=x,
                                         y=y,
                                         verbose=verbose)
        print(str(indexModel)+' Model results:')
        print('Test loss:', self.score[0])
        print('Test accuracy:', self.score[1])

    def Save_weights(self, filepath):
        self.model.save_weights(filepath)

    def Load_weights(self, filepath):
        self.model.load_weights(filepath)

    def SetWeights(self, indexLayer):
        self.model.get_layer(index=indexLayer).set_weights(copy.deepcopy(self.weights.pop(0)))

    def GetWeights(self, indexLayer):
        # Copy the weighs from the given model
        #self.weights = copy.deepcopy(self.model.get_layer(index=indexLayer).get_weights())
        self.weights.append(copy.deepcopy(self.model.get_layer(index=indexLayer).get_weights()))

    def ClearWeights(self):
        del self.weights[:]

    def SaveWeightsInShadow(self):
        self.weightsShadow = copy.deepcopy(self.weights)

    def LoadWeightsFromShadow(self):
        self.weights = copy.deepcopy(self.weightsShadow)

    def SaveWeightsIn1stModel(self):
        #self.weights1stModel = copy.deepcopy(self.model.get_layer(index=indexLayer).get_weights())
        for i in range(self.numLayer):
            self.weights1stModel.append(copy.deepcopy(self.model.get_layer(index=i+1).get_weights()))

    def LoadWeightsFrom1stModel(self):
        for i in range(self.numLayer):
            self.model.get_layer(index=i+1).set_weights(copy.deepcopy(self.weights1stModel.pop(0)))

    def SetCntPossibleIteration(self, x):
        self.cntPossibleIteration = x

    def GetCntPossibleIteration(self):
        return self.cntPossibleIteration

    def OptimizeNetwork(self, testNumber, titleLargeEpochWeight, titleSmallEpochWeight, callBackFunction,
                        cntIter=1, tupleLayer=(1, ), x_train=0, y_train=0, x_test=0, y_test=0, epochs=1, batch_size=128):
        cntIteration = 0
        bIteration = True
        bRetraining = True

        # Determine the listIndexNonOutliers and listIndexOutliers from the weights of Large Epoch
        self.Load_weights(titleLargeEpochWeight)
        self.ClearWeights()
        for e in tupleLayer:
            self.GetWeights(e)
        self.InitializeIndex(tupleLayer, testNumber, cntIter)

        # Get the weights from Small Epoch
        self.Load_weights(titleSmallEpochWeight)
        self.ClearWeights()
        for e in tupleLayer:
            self.GetWeights(e)

        # Set as the 2nd model
        self.SetId(2)

        # Initialize the counter
        global_variables.cntEpochs = 0

        # Save the weights of 1st model in the shadow
        self.SaveWeightsInShadow()

        while (bIteration):
            cntIteration += 1
            # Alter the weights that will be used as the initialization values of the model
            self.ReplaceOutliers(testNumber)

            # Set weights on the model
            for e in tupleLayer:
                self.SetWeights(e)

            # Retraining the model
            self.Compile(loss=keras.losses.mse,
                         optimizer=keras.optimizers.Adadelta(),
                         metrics=['accuracy'])

            self.Fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=0,
                     callbacks=[callBackFunction.WeightScale()],
                     validation_data=(x_test, y_test))
            self.Load_weights('../results/#Epoch' + str(global_variables.cntEpochs) +
                              '_weights_of_2nd_model_' + str(testNumber) + '.h5')

            # Evalute the model
            self.Evaluate(x_test[:500], y_test[:500], verbose=0, indexModel=2)
            self.Evaluate(x_test[:107], y_test[:107], verbose=0, indexModel=2)

            # Get weights from the model
            for e in tupleLayer:
                self.GetWeights(e)

            # Plot the weights after retraining
            print("Plot the weights after retraining")
            for i in range(self.numWeightsLayer):
                self.PlotCurrentWeights(i,
                                        str(i+1)+' layer weights_'+'#' + str(cntIteration) + ' Plots of weights after retraining_' + str(testNumber),
                                        '../results/'+str(i+1)+' layer weights_'+'#' + str(cntIteration) + ' Weights after retraining_' + str(testNumber) +'.html')

            # Determine whether it will iterate(and/or retrain) more or not
            bIteration, bRetraining = self.DetermineIteration()

            if(bRetraining):
                # Load the weights from the shadow
                # i.e. This is the step to prepare the further retraining.
                self.LoadWeightsFromShadow()
            else:
                # Load the weights from the 1st model
                # i.e. It turns out that retraining(optimization) is not valid.
                self.LoadWeightsFrom1stModel()


    def InitializeIndex(self, tupleLayer, testNumber, cntIter):
        # Initialize the dimensions of the weights
        #self.numWeightsLayer = int(len(self.weights)/ len(self.weights[0]))
        self.numWeightsLayer = len(self.weights)

        # Initialize the variables
        for i in range(self.numWeightsLayer):
            valShape = self.weights[i][0].shape
            self.numRow.append(valShape[0])
            self.numCol.append(valShape[1])
            self.numInputSlices.append(valShape[2])
            self.numOutputSlices.append(valShape[3])

        # Initialize the list
        self.listIndexOutliers = [[] for i in range(self.numWeightsLayer)]
        self.listIndexNonOutliers = [[] for i in range(self.numWeightsLayer)]
        #listCnt = [0 for i in range(self.numOutputSlices)]
        listCnt = [0 for i in range(self.numWeightsLayer)]
        listIndexZero = [[] for i in range(self.numWeightsLayer)]
        listIndexNonZero = [[] for i in range(self.numWeightsLayer)]

        # Plot the weights before replacing
        print("Plot the weights of 1st model")
        for i in range(self.numWeightsLayer):
            self.PlotCurrentWeights(i,
                                    'Plots of weights of 1st model_' + str(i+1) + ' layer weights_' + str(testNumber),
                                    '../results/Weights of 1st model_' + str(i+1) + ' layer weights_'  + str(testNumber) + '.html')

        # Save the weights into weights1stModel
        self.SaveWeightsIn1stModel()

        # Step 3: Find non-sparse solution
        for i in range(self.numWeightsLayer):
            print(str(self.numWeightsLayer + 1) + "layer's weights information")

            # Step 3.1.1: Find the number of non-zero solutions in each set of weights
            listCnt[i] = self.FindNumberNonZero(i)
            print("listCnt with all-zero weights: " + str(listCnt[i]))

            # Step 3.1.2: Find the avg value out of the listCnt
            self.listAvg.append(statistics.mean(listCnt[i]))

            # Step 3.1.3: Remove all-zero weights from the listCnt
            self.RemoveAllzeroWeights(listCnt[i], listIndexZero[i], listIndexNonZero[i])
            print("listCnt without all-zero weights : "+str(listCnt[i]))

            # Step 3.2: Calculate the mean and variance of the numbers
            self.mean.append(np.mean(listCnt[i]))
            self.stdDeviation.append(np.std(listCnt[i]))

            # Step 3.3: Find outliers which are far from the zero after rescaling using z-score
            for outputSlices in range(len(listCnt[i])):
                listCnt[i][outputSlices] = (listCnt[i][outputSlices] - self.mean[i]) / (self.stdDeviation[i])
                if (listCnt[i][outputSlices] > 0.5): # 0.5 represents Top 30% in the normalized distribution
                    self.listIndexOutliers[i].append(listIndexNonZero[i][outputSlices])
                else:
                    self.listIndexNonOutliers[i].append(listIndexNonZero[i][outputSlices])
            print("Index of Outliers: " + str(self.listIndexOutliers[i]))
            print("Index of NonOutliers: " + str(self.listIndexNonOutliers[i]))

        # Count the number of NonOutliers
        #self.cntPossibleIteration = len(self.listIndexNonOutliers)

        # It will iterate up to the defined counts
        self.SetCntPossibleIteration(cntIter)

    def ReplaceOutliers(self, testNumber):
        print("Remained iteration: " + str(self.cntPossibleIteration))
        self.cntReplaceOutliers += 1

        # Step 4: Replace outliers by non-outliers which are randomly selected
        for n in range(self.numWeightsLayer):
            # Define the buffer of weights
            tempListWeights = [[[[] for i in range(self.numInputSlices[n])] for i in range(self.numCol[n])] for i in range(self.numRow[n])]

            # Plot the weights before replacing
            print("Plot the weights before replacing")
            self.PlotCurrentWeights(n,
                                    str(n+1) + ' layer weights_'+'#' + str(self.cntReplaceOutliers) + ' Plots of weights before replacing_' + str(testNumber),
                                    '../results/' + str(n+1) + ' layer weights_'+'#' + str(self.cntReplaceOutliers) + ' Weights before replacing_' + str(testNumber) + '.html')

            # Replace the weights by the one that comes from Non-outliers
            for l in range(len(self.listIndexOutliers[n])):
                # Copy the sparse weights into the buffer
                # Dimension: (row, col, inputSlice, outputSlice)
                k = random.choice(self.listIndexNonOutliers[n])
                print("index of selected non-outlier:" + str(k))
                for i in range(self.numRow[n]):
                    for j in range(self.numCol[n]):
                        for s in range(self.numInputSlices[n]):
                            tempListWeights[i][j][s] = self.weights[n][0][i][j][s][k]

                # Flatten the list using a nested list comprehension
                listWeightsShuffled = [w for sublist1 in tempListWeights for sublist2 in sublist1 for w in sublist2]

                # Permute and change the sign of the weights in the list
                listWeightsShuffled = np.negative(np.random.permutation(listWeightsShuffled))

                # Insert shuffled weights into the outliers
                k = self.listIndexOutliers[n][l]
                print("index of selected outlier:" + str(k))
                for i in range(self.numRow[n]):
                    for j in range(self.numCol[n]):
                        for s in range(self.numInputSlices[n]):
                            self.weights[n][0][i][j][s][k] = listWeightsShuffled[i*self.numCol[n]*self.numInputSlices[n] + j*self.numInputSlices[n] + s]
                print("The outlier has been replaced by the non-outlier")

                # Plot the weights after replacing
                print("Plot the weights after replacing")
                self.PlotCurrentWeights(n,
                                        str(n+1) + ' layer weights_'+'#' + str(self.cntReplaceOutliers) + ' Plots of weights after replacing_' + str(testNumber),
                                        '../results/' + str(n+1) + ' layer weights_'+'#' + str(self.cntReplaceOutliers) + ' Weights after replacing_' + str(testNumber) + '.html')

    def DetermineIteration(self):
        bIteration = False  # It indicates whether we need further retraining or not
        bRetraining = True  # It represents the validity of whole retraining process

        # Initialize the lists
        listCnt = [0 for i in range(self.numWeightsLayer)]
        listIndexZero = [[] for i in range(self.numWeightsLayer)]
        listIndexNonZero = [[] for i in range(self.numWeightsLayer)]
        listAvg = []
        listBoolIteration = []
        listBoolRetraining = []

        # Find the max value
        for i in range(self.numWeightsLayer):
            # Step 3.1.1: Find the number of non-zero solutions in each set of weights
            listCnt[i] = self.FindNumberNonZero(i)
            print("listCnt with all-zero weights: "+str(listCnt[i]))

            # Step 3.1.2: Find the avg value out of the listCnt
            listAvg.append(statistics.mean(listCnt[i]))

            # Step 3.1.3: Remove all-zero weights from the listCnt
            self.RemoveAllzeroWeights(listCnt[i], listIndexZero[i], listIndexNonZero[i])
            print("listCnt without all-zero weights : "+str(listCnt[i]))


        # Compare the max values
        self.cntPossibleIteration -= 1
        for i in range(self.numWeightsLayer):
            # If the avg value is equal or larger than before
            if (listAvg[i] >= self.listAvg[i]):
                # If it has iterated as many times as the number of NonOutliers
                if (self.cntPossibleIteration == 0):
                    listBoolIteration.append(False)
                    listBoolRetraining.append(False)
                else:
                    listBoolIteration.append(True)
            else:
                listBoolIteration.append(False)

        # Assign the value on bIteration and bRetraining
        if True in listBoolIteration:
            print("Retrain again")
            bIteration = True

        if False in listBoolRetraining:
            print("It already retrained as many time as the number you defined")
            print("Retraining didn't work. 1st model is loaded back.")
            bRetraining = False
            bIteration = False

        if (bIteration == False) and (bRetraining == True):
            print("Retraining has been finished")

        return bIteration, bRetraining

    def FindNumberNonZero(self, indexLayer):
        # Step 3.1.1: Find the number of non-zero solutions in each set of weights
        listCnt = [0 for i in range(self.numOutputSlices[indexLayer])]
        for outputSlices in range(self.numOutputSlices[indexLayer]):
            for inputSlices in range(self.numInputSlices[indexLayer]):
                for row in range(self.numRow[indexLayer]):
                    for col in range(self.numCol[indexLayer]):
                        if (abs(self.weights[indexLayer][0][row][col][inputSlices][outputSlices]) > 0.01):
                            listCnt[outputSlices] += 1

        return copy.deepcopy(listCnt)

    def RemoveAllzeroWeights(self, listCnt, listIndexZero, listIndexNonZero):
        # Step 3.1.2: Remove all-zero weights from the listCnt
        listCntCopied = listCnt[:]
        for i in range(len(listCntCopied)):  # Python doesn't update the counter of listCopied during the iteration
            if (listCntCopied[i] == 0):
                listIndexZero.append(i)
                listCnt.remove(0)
            else:
                listIndexNonZero.append(i)

    def PlotCurrentWeights(self, indexLayer, title, filename):
        values = [i + 1 for i in range(self.numRow[indexLayer] * self.numCol[indexLayer] * self.numInputSlices[indexLayer])]
        weightsSorted = self.weights[indexLayer][0].reshape(self.numRow[indexLayer] * self.numCol[indexLayer] *
                                                            self.numInputSlices[indexLayer] * self.numOutputSlices[indexLayer])
        weightsSorted = weightsSorted.reshape((self.numOutputSlices[indexLayer],
                                               self.numRow[indexLayer] * self.numCol[indexLayer] * self.numInputSlices[indexLayer]), order='F')
        HOUtils().PlotWeights(values, weightsSorted, title, filename)