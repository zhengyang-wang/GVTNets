import tensorflow as tf
from .basic_ops import *

def get_loss(labels, inputs, outputs, loss_type, probalistic, offset, dimension):
	"""
	Args:
		labels: The ground truth output tensor, same dimensions as 'inputs' and 'outputs'.
		inputs: The input tensor (source images) of UNet.
		outputs: Output tensor of UNet.
		loss_type: String. Type of loss function. Can be either 'MSE' (Mean Squared Error) or 'MAE' (Mean Absolute Error).
		probalistic: Boolean. Whether to use probalistic loss.
		offset: Boolean. Whether to add UNet inputs to outputs.
		dimension: String. Dimension of image, '3D' or '2D'.
	"""

	if dimension == '2D':
		convolution = convolution_2D
	elif dimension == '3D':
		convolution = convolution_3D

	if loss_type == 'MSE':
		if probalistic:
			preds = tf.add(inputs, outputs) if offset else outputs
			sigma = convolution(outputs, 1, 1, 1, False, name = 'out_sigma_conv')
			sigma = tf.nn.softplus(sigma) + 1e-3
			loss = tf.reduce_mean(tf.truediv(tf.square(preds-labels), sigma) + tf.log(sigma))
		else:
			preds = tf.add(inputs, outputs) if offset else outputs
			loss = tf.losses.mean_squared_error(labels, preds)

	elif loss_type == 'MAE':
		if probalistic:
			preds = tf.add(inputs, outputs) if offset else outputs
			sigma = convolution(outputs, 1, 1, 1, False, name = 'out_sigma_conv')
			sigma = tf.nn.softplus(sigma) + 1e-3
			loss = tf.reduce_mean(tf.truediv(tf.abs(preds-labels), sigma) + tf.log(sigma))
		else:
			preds = tf.add(inputs, outputs) if offset else outputs
			loss = tf.losses.absolute_difference(labels, preds)

	else:
		raise ValueError("The opts.loss_type (%s) must be MSE or MAE." % (opts.loss_type))

	return loss
