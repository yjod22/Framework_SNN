#
###############################################################################
#                                                                             #
#							 Copyright (c)									  #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:     utils.py
#
###############################################################################
#  Description:
#
#  (For a detailed description look at the object description in the UML model)
#
###############################################################################
# History
################################################################################
# File:		   utils.py
# Version:     8.0
# Author/Date: Junseok Oh / 2019-05-23
# Change:      (SCR_V6.4-1): NN Optimization-JSO (Make use of listIndex not to consider zero weights in addition)
#			   (SCR_V6.4-4): Create SaveInTxtFormat function
#			   (SCR_V6.4-12): Create GetConvolutionLayerWeightsBiasesSN for adaption 
#			   (SCR_V6.4-16): Fix bug of weight ordering (weights[j, i, k, l])
#			   (SCR_V6.4-19): Make use of the plotly
#			   (SCR_V6.4-31): Exception handling when the number is out of range (-1, +1)
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
from functools import reduce
import operator
from plotly import tools
import plotly as py
import plotly.graph_objs as go

class HOUtils(object):
	def __init__(self):
		pass

	def createSN(self, x, length):
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
					print("x: "+ str(x))
		else:
			for i in range(int(np.floor(((x + 1) / 2) * length))):
				try:
					x_SN[i] = True
				except IndexError:
					print("The number is out of range (-1, +1)")
					print("x: "+ str(x))
		np.random.shuffle(x_SN)
		return x_SN


	def GetConvolutionLayerWeightsBiasesSN(self, model, indexLayer, length, **kwargs):
		# Select the activation function to use
		bAdaptive = "False"
		for key in kwargs:
			if (key == "Adaptive"):
				bAdaptive = kwargs[key]

		# Extract weights of convolution layer from model
		weights = model.get_layer(index=indexLayer).get_weights()[0]
		# Extract biases of convolution layer from model
		bBias = True
		try :
			biases = model.get_layer(index=indexLayer).get_weights()[1]
		except IndexError:
			bBias = False

		# Create variables for stochastic number of the weights
		width = int(weights.size / weights[0].size)  # 5
		height = int(weights[0].size / weights[0][0].size)  # 5
		inputSlices = int(weights[0][0].size / weights[0][0][0].size)  # 20
		outputSlices = int(weights[0][0][0].size / weights[0][0][0][0].size)  # 50
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
						weight_SNs[i, j, k, l] = self.createSN(res_weights[i, j, k, l], length)

						if(bAdaptive == "True"):
							# Make the list of Indices which indicate the positions of non-near-zero elements
							if (np.abs(res_weights[i, j, k, l]) > 0.01):
								listIndex[i].append(j * height * width + k * width + l)
						else:
							# Generate the list of Indices
							listIndex[i].append(j * height * width + k * width + l)
			if(bBias):
				# Generate the stochastic numbers of the biases
				bias_SNs[i] = self.createSN(biases[i], length)

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

	def GetConnectedLayerWeightsSN(self, model, indexLayer, length):
		# Extract weights of fully connected layer from model
		dense_weights = model.get_layer(index=indexLayer).get_weights()[0]

		# Create a variable for stochastic number of the weights
		tensors = int(dense_weights.size / dense_weights[0].size)
		classes = int(dense_weights[0].size / dense_weights[0][0].size)
		dense_weight_SNs = np.full((tensors, classes, length), False)

		# Create a stochastic number of the weights
		for i in range(tensors):
			for j in range(classes):
				dense_weight_SNs[i, j] = self.createSN(dense_weights[i, j], length)

		return dense_weight_SNs

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

	def stochtoint(self, x):
		"""convert bipolar stochastic number to integer"""
		return (sum(x) / len(x)) * 2.0 - 1.0

	def SaveInTxtFormat(self, title, testIndex, outputMatrix, inputSlices, row, col, layerNModel, xTest):
		# Convert Stochastic number to Binary number
		conv_out_test = np.zeros((inputSlices, row, col))
		for i in range(inputSlices):
			for j in range(row):
				for k in range(col):
					conv_out_test[i, j, k] = self.stochtoint(outputMatrix[i, j, k])

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