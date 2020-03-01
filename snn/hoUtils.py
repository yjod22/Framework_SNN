###############################################################################
#                                                                             #
#                            Copyright (c)                                    #
#                         All rights reserved.                                #
#                                                                             #
###############################################################################
#
#  Filename:	hoUtils.py
#  Description:	
#  Author/Date:	Junseok Oh / 2020-02-27
#  Initiator:	Florian Neugebauer
################################################################################

import numpy as np
from plotly import tools
import plotly as py
import plotly.graph_objs as go
from snn.hoSnn import HOSnn

"""
Architecture of the classes
  HOSnn
    |		
 HOUtils   
"""


class HOUtils(HOSnn):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def GetConvolutionLayerWeightsBiasesSN(self, model, indexLayer, **kwargs):
		"""
		Extracting weights and biases of a convolution layer from a trained binary model
		Extracted weights and biases are converted to stochastic numbers
		
		Parameters
		----------
		model: object
			the trained binary model which consists of layers
		
		indexLayer: int
			the index of the convolution layer from which weights and biases would be extracted
		
		Adaptive: string
			the input size of a MUX can be reduced as much as it has near-zero inputs
			Possible elements: "True", "False"
		
		Returns
		-------
		weight_SNs: object
			the weights of the convolution layer in stochastic numbers
		
		bias_SNs: object
			the biases of the convolution layer in stochastic numbers
		
		listIndex: object (two-dimensional list)
			the list of indices which indicate the positions of non-near-zero weights
			e.g. [ [1, 3, 4], [2, 5] ]  indicates the followings:
				# the number of outputs is two
				# 1st, 3rd, and 4th weight of the first output are not zero
				# 2nd and 5th weight of the second output are not zero
		"""
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
		"""
		Extracting biases of a fully-connected layer from a trained binary model
		Extracted weights and biases are binary numbers
		
		Parameters
		----------
		model: object
			the trained binary model which consists of layers
		
		indexLayer: int
			the index of the fully-connected layer from which weights and biases would be extracted
				
		Returns
		-------
		bias_SNs: object
			the biases of the fully-connected layer in binary numbers		
		"""
		dense_biases = model.get_layer(index=indexLayer).get_weights()[1]
		return dense_biases

	def GetConnectedLayerWeights(self, model, indexLayer):	
		"""
		Extracting biases of a fully-connected layer from a trained binary model
		Extracted weights and biases are binary numbers
		
		Parameters
		----------
		model: object
			the trained binary model which consists of layers
		
		indexLayer: int
			the index of the fully-connected layer from which weights and biases would be extracted
				
		Returns
		-------
		bias_SNs: object
			the biases of the fully-connected layer in binary numbers		
		"""
		dense_weights = model.get_layer(index=indexLayer).get_weights()[0]
		return dense_weights

	def GetConnectedLayerWeightsBiasesSN(self, model, indexLayer, **kwargs):
		"""
		Extracting weights and biases of a fully-connected layer from a trained binary model
		Extracted weights and biases are converted to stochastic numbers
		
		Parameters
		----------
		model: object
			the trained binary model which consists of layers
		
		indexLayer: int
			the index of the fully-connected layer from which weights and biases would be extracted
		
		Adaptive: string
			the input size of a MUX can be reduced as much as it has near-zero inputs
			Possible elements: "True", "False"
		
		Returns
		-------
		dense_weight_SNs: object
			the weights of the fully-connected layer in stochastic numbers
		
		dense_bias_SNs: object
			the biases of the fully-connected layer in stochastic numbers
		
		listIndexDense: object (two-dimensional list)
			the list of indices which indicate the positions of non-near-zero weights
			e.g. [ [1, 3, 4], [2, 5] ]  indicates the followings:
				# the number of classes is two
				# 1st, 3rd, and 4th weight of the first class are not zero
				# 2nd and 5th weight of the second class are not zero
		"""
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
		listIndexDense = [[] for i in range(classes)]

		# Create a stochastic number of the weights
		for j in range(classes):
			for i in range(tensors):
				dense_weight_SNs[i, j] = self.CreateSN(dense_weights[i, j])

				if(bAdaptive == "True"):
					# Make the list of Indices which indicate the positions of non-near-zero elements
					if (np.abs(dense_weights[i, j]) > 0.01):
						listIndexDense[j].append(i)
				else:
					listIndexDense[j].append(i)

			if (bBias):
				# Generate the stochastic numbers of the biases
				dense_bias_SNs[j] = self.CreateSN(biases[j])

				# Generate the index of the bias in the list at the end
				listIndexDense[j].append(tensors)

		return dense_weight_SNs, dense_bias_SNs, listIndexDense

	def BinaryConnectedLayer(self, numTensors, numClasses, dense_input, dense_weights, dense_biases):
		"""
		Performing convolution operations in a fully-connected layer not using stochastic numbers
		
		Parameters
		----------
		numTensors: int
			the number of the fully-connected layer's inputs
			
		numClasses: int
			the number of the fully-connected layer's outputs
			
		dense_input: object
			inputs of the fully-connected layer over which a kernel would perform its convolution operation
			
		dense_weights: object
			the fully-connected layer's weights that are extracted by GetConnectedLayerWeights function
			
		dense_biases: object
			the fully-connected layer's biases that are extracted by GetConnectedLayerBiases function
			
		Returns
		-------
		dense_output: object
			outputs of the fully-connected layer
		"""
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
		"""
		Saving the intermediate results of a convolution layer in a txt file
		
		Parameters
		----------
		title: string
			the title of the txt file
		
		testIndex: int
			the index of a test sample
			
		outputMatrix: object
			the output of the stochastic convolution layer
			
		inputSlices: int
			the number of input slices of the convolution layer
			
		row: int
			the vertical size of the convolution layer's kernel
		
		col: int
			the horizontal size of the convolution layer's kernel
			
		layerNModel: object
			the output of the binary convolution layer
			
		xTest: object
			the inputs of the test sample
		"""
	
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
		"""
		Plotting given weights in a html file
		
		Parameters
		----------
		values: object
			the x-axis data of the plot, the indices of the weights
			
		weightsSorted: object
			the y-axis data of the plot, the sizes of the weights
		
		title: string
			the title of the plot
		
		filename: string
			the name of the html file
		"""
		numOutputSlices = int(weightsSorted.size / weightsSorted[0].size)
		trace = []
		fig = tools.make_subplots(rows=numOutputSlices, cols=1)
		for i in range(numOutputSlices):
			trace.append(go.Bar(x=values, y=weightsSorted[i]))
			fig.append_trace(trace[i], i + 1, 1)
		fig['layout'].update(title=title)
		py.offline.plot(fig, filename=filename, auto_open=False)