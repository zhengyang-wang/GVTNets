import tensorflow as tf
from .basic_ops import *


def get_loss(labels, inputs, outputs, loss_type, probalistic, dimension):

	if dimension == '2D':
		convolution = convolution_2D
	elif dimension == '3D':
		convolution = convolution_3D

	if loss_type == 'MSE':
		if probalistic:
			preds = tf.add(inputs, outputs)
			sigma = convolution(outputs, 1, 1, 1, False, name = 'out_sigma_conv')
			sigma = tf.nn.softplus(sigma) + 1e-3
			loss = tf.reduce_mean(tf.truediv(tf.square(preds-labels), sigma) + tf.log(sigma))
		else:
			loss = tf.losses.mean_squared_error(labels, outputs)

	elif loss_type == 'MAE':
		if probalistic:
			preds = tf.add(inputs, outputs)
			sigma = convolution(outputs, 1, 1, 1, False, name = 'out_sigma_conv')
			sigma = tf.nn.softplus(sigma) + 1e-3
			loss = tf.reduce_mean(tf.truediv(tf.abs(preds-labels), sigma) + tf.log(sigma))
		else:
			loss = tf.losses.absolute_difference(labels, outputs)

	else:
		raise ValueError("The loss_type (%s) must be MSE or MAE." % (loss_type))

	return loss