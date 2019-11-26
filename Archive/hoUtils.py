#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     hoUtils.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   hoUtils.py
# Version:     15.0
# Author/Date: Junseok Oh / 2019-07-01
# Change:      (SCR_V14.0-1): Modularize the classes, change the file names
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   utils.py
# Version:     14.0
# Author/Date: Junseok Oh / 2019-07-01
# Change:      (SCR_V13.0-1): Place CreateSN on the higher class
#              (SCR_V13.0-2): Place StochToInt on the higher class
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   utils.py
# Version:     8.0
# Author/Date: Junseok Oh / 2019-05-23
# Change:      (SCR_V6.4-1): NN Optimization-JSO (Make use of listIndex not to consider zero weights in addition)
#              (SCR_V6.4-4): Create SaveInTxtFormat function
#              (SCR_V6.4-12): Create GetConvolutionLayerWeightsBiasesSN for adaption
#              (SCR_V6.4-16): Fix bug of weight ordering (weights[j, i, k, l])
#              (SCR_V6.4-19): Make use of the plotly
#              (SCR_V6.4-31): Exception handling when the number is out of range (-1, +1)
# Cause:       -
# Initiator:   Florian Neugebauer
################################################################################
# File:		   utils.py
# Version:     5.4 (SCR_V5.3-6)
# Author/Date: Junseok Oh / 2018-11-27
# Change:      Create functions
# Cause:       Different NN need same functions for extracting weights, biases
# Initiator:   Florian Neugebauer
################################################################################

import numpy as np
from plotly import tools
import plotly as py
import plotly.graph_objs as go
from snn.hoSnn import HOSnn

class HOUtils(HOSnn):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def GetConvolutionLayerWeightsBiasesSN(self, model, indexLayer, **kwargs):
		# Select the activation function to use
		bAdaptive = "False"
		for key in kwargs:
			if (key == "Adaptive"):
				bAdaptive = kwargs[key]

		# Set the length of stochastic number
		length = self.snLength

		# Extract weights of convolution layer from model
		weights = model.get_layer(index=indexLayer).get_weights()[0]
		# Extract biases of convolution layer from model
		bBias = True
		try :
			biases = model.get_layer(index=indexLayer).get_weights()[1]
		except IndexError:
			bBias = False

		# Create variables for stochastic number of the weights
		width, height, inputSlices, outputSlices = weights.shape
		weight_SNs = np.full((outputSlices, inputSlices, height, width, length), False)

		# Create a variable for stochastic number of the biases
		bias_SNs = np.full((outputSlices, length), False)

		# Create variables for stochastic number of the biases, weights and its maps(lists)
		res_weights = np.zeros((outputSlices, inputSlices, height, width))
		listIndex = [[] for i in range(outputSlices)]

		# Change the order as the one is used in SNN (outputSlices, inputSlices, row, col)
		# The order in BNN is as follows (row, col, inputSlices, outputSlices)
		# row is height, col is width
		for i in range(width): # col
			for j in range(height): # row
				for k in range(inputSlices):
					for l in range(outputSlices):
						res_weights[l, k, j, i] = weights[j, i, k, l]

		# Generate the stochastic numbers of the biases, weights and its maps(lists)
		for i in range(outputSlices):
			for j in range(inputSlices):
				for k in range(height):
					for l in range(width):
						# Generate the stochastic numbers of the weights
						weight_SNs[i, j, k, l] = self.CreateSN(res_weights[i, j, k, l])

						if(bAdaptive == "True"):
							# Make the list of Indices which indicate the positions of non-near-zero elements
							if (np.abs(res_weights[i, j, k, l]) > 0.01):
								listIndex[i].append(j * height * width + k * width + l)
						else:
							# Generate the list of Indices
							listIndex[i].append(j * height * width + k * width + l)
			if(bBias):
				# Generate the stochastic numbers of the biases
				bias_SNs[i] = self.CreateSN(biases[i])

				# Generate the index of the bias in the list at the end
				listIndex[i].append(width*height*inputSlices)

		# Return the variable
		return weight_SNs, bias_SNs, listIndex


	def GetConnectedLayerBiases(self, model, indexLayer):
		dense_biases = model.get_layer(index=indexLayer).get_weights()[1]
		return dense_biases

	def GetConnectedLayerWeights(self, model, indexLayer):
		dense_weights = model.get_layer(index=indexLayer).get_weights()[0]
		return dense_weights

	def GetConnectedLayerWeightsSN(self, model, indexLayer):
		# Extract weights of fully connected layer from model
		dense_weights = model.get_layer(index=indexLayer).get_weights()[0]

		# Set the length of stochastic number
		length = self.snLength

		# Create a variable for stochastic number of the weights
		tensors, classes = dense_weights.shape
		dense_weight_SNs = np.full((tensors, classes, length), False)

		# Create a stochastic number of the weights
		for i in range(tensors):
			for j in range(classes):
				dense_weight_SNs[i, j] = self.CreateSN(dense_weights[i, j])

		return dense_weight_SNs

	def GetConnectedLayerWeightsBiasesSN(self, model, indexLayer, **kwargs):
		# Select the activation function to use
		bAdaptive = "False"
		for key in kwargs:
			if (key == "Adaptive"):
				bAdaptive = kwargs[key]

		# Set the length of stochastic number
		length = self.snLength

		# Extract weights of fully connected layer from model
		dense_weights = model.get_layer(index=indexLayer).get_weights()[0]
		# Extract biases of convolution layer from model
		bBias = True
		try:
			biases = model.get_layer(index=indexLayer).get_weights()[1]
		except IndexError:
			bBias = False

		# Create a variable for stochastic number of the weights
		tensors, classes = dense_weights.shape
		dense_weight_SNs = np.full((tensors, classes, length), False)

		# Create a variable for stochastic number of the biases
		dense_bias_SNs = np.full((classes, length), False)

		# Create a variable for the maps(lists)
		listIndex = [[] for i in range(classes)]

		# Create a stochastic number of the weights
		for j in range(classes):
			for i in range(tensors):
				dense_weight_SNs[i, j] = self.CreateSN(dense_weights[i, j])

				if(bAdaptive == "True"):
					# Make the list of Indices which indicate the positions of non-near-zero elements
					if (np.abs(dense_weights[i, j]) > 0.01):
						listIndex[j].append(i)
				else:
					listIndex[j].append(i)

			if (bBias):
				# Generate the stochastic numbers of the biases
				dense_bias_SNs[j] = self.CreateSN(biases[j])

				# Generate the index of the bias in the list at the end
				listIndex[j].append(tensors)

		return dense_weight_SNs, dense_bias_SNs, listIndex

	def BinaryConnectedLAyer(self, numTensors, numClasses, dense_input, dense_weights, dense_biases):
		# Conventional binary function for the fully connected layer
		# 1. Set the dense_output_res
		dense_output_res = np.zeros((numTensors, numClasses))  # num_last_classes=10

		# 2. Perform the convolution operations from the dense_output
		for j in range(numClasses):
			for i in range(numTensors):
				dense_output_res[i, j] = (dense_input[0, i]) * (dense_weights[i, j])

		# 3. Set the dense_output
		dense_output = np.zeros((1, numClasses))

		# 4. Sum up the results of convolution operations over all tensors
		for j in range(numClasses):
			for i in range(numTensors):
				dense_output[0, j] = dense_output[0, j] + dense_output_res[i, j]

		# 5. Perform the biases addition
		dense_output = dense_output + dense_biases

		return dense_output


	def SaveInTxtFormat(self, title, testIndex, outputMatrix, inputSlices, row, col, layerNModel, xTest):
		# Convert Stochastic number to Binary number
		conv_out_test = np.zeros((inputSlices, row, col))
		for i in range(inputSlices):
			for j in range(row):
				for k in range(col):
					conv_out_test[i, j, k] = self.StochToInt(outputMatrix[i, j, k])

		# Predict the intermediate results from the Binary Neural Network
		BNN_prediction = layerNModel.predict(np.asarray([xTest[testIndex]]))

		# Write the array to disk
		txtTitle = title + '_SNN_' + str(testIndex + 1) + '.txt'
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
		del(conv_out_test)

		txtTitle = title + '_BNN_' + str(testIndex + 1) + '.txt'
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

	def PlotWeights(self, values, weightsSorted, title, filename):
		numOutputSlices = int(weightsSorted.size / weightsSorted[0].size)
		trace = []
		fig = tools.make_subplots(rows=numOutputSlices, cols=1)
		for i in range(numOutputSlices):
			trace.append(go.Bar(x=values, y=weightsSorted[i]))
			fig.append_trace(trace[i], i + 1, 1)
		fig['layout'].update(title=title)
		py.offline.plot(fig, filename=filename)