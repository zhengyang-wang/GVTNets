import tensorflow as tf
from .basic_ops import *

def get_loss(labels, outputs, opts, conf_unet):

	if conf_unet['dimension'] == '2D':
		convolution = convolution_2D
	elif conf_unet['dimension'] == '3D':
		convolution = convolution_3D

	if self.opts.loss_type == 'MSE':
		if self.opts.probalistic:
            sigma = convolution(out, 1, 1, 1, False, name = 'out_sigma_conv')
            sigma = tf.nn.softplus(sigma) + 1e-3
            loss = tf.reduce_mean(tf.truediv(tf.square(preds-labels), sigma) + tf.log(sigma))
		else:
			loss = tf.losses.mean_squared_error(labels, outputs)

	elif self.opts.loss_type == 'MAE':
		if self.opts.probalistic:
            sigma = convolution(out, 1, 1, 1, False, name = 'out_sigma_conv')
            sigma = tf.nn.softplus(sigma) + 1e-3
            loss = tf.reduce_mean(tf.truediv(tf.abs(preds-labels), sigma) + tf.log(sigma))
        else:
			loss = tf.losses.absolute_difference(labels, outputs)

	return loss