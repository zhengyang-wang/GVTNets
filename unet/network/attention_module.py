import tensorflow as tf
import sys
sys.path.append('../..')
from network_configure import conf_attn_same, conf_attn_up, conf_attn_down
from .basic_ops import *
from .attention_layer import self_attention


"""This script defines the attention modules: same-, up-, down- global transformer operators.
Note that pre-activation is used for residual-like blocks.
"""


def same_gto(inputs, output_filters, training, dimension, name):
	"""Same GTO block.

	Args:
		inputs: a Tensor with shape [batch, (d,) h, w, channels]
		output_filters: an integer
		training: a boolean for batch normalization and dropout
		dimension: a string, dimension of inputs/outputs -- 2D, 3D
		name: a string

	Returns:
		A Tensor of shape [batch, (_d,) _h, _w, output_filters]
	"""
	with tf.variable_scope(name):
		shortcut = inputs
		inputs = batch_norm(inputs, training, 'batch_norm')
		inputs = relu(inputs, 'relu')
		inputs, _ = self_attention(
						inputs,
						output_filters // conf_attn_same['key_ratio'],
						output_filters // conf_attn_same['value_ratio'],
						output_filters,
						conf_attn_same['num_heads'],
						training,
						dimension,
						'SAME',
						'attention',
						conf_attn_same['dropout_rate'],
						conf_attn_same['use_softmax'],
						conf_attn_same['use_bias'])
		return tf.add(shortcut, inputs)


def up_gto_v1(inputs, output_filters, training, dimension, name):
	"""Up GTO block version 1."""
	if dimension == '2D':
		projection_shortcut = transposed_convolution_2D
	elif dimension == '3D':
		projection_shortcut = transposed_convolution_3D
	else:
		raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

	with tf.variable_scope(name):
		# The projection_shortcut should come after the batch norm and ReLU.
		inputs = batch_norm(inputs, training, 'batch_norm')
		inputs = relu(inputs, 'relu')
		shortcut = projection_shortcut(inputs, output_filters, 3, 2, False, 'projection_shortcut')
		inputs, _ = self_attention(
						inputs,
						output_filters // conf_attn_up['key_ratio'],
						output_filters // conf_attn_up['value_ratio'],
						output_filters,
						conf_attn_up['num_heads'],
						training,
						dimension,
						'UP',
						'attention',
						conf_attn_up['dropout_rate'],
						conf_attn_up['use_softmax'],
						conf_attn_up['use_bias'])
		return tf.add(shortcut, inputs)


def down_gto_v1(inputs, output_filters, training, dimension, name):
	"""Down GTO block version 1."""
	if dimension == '2D':
		projection_shortcut = convolution_2D
	elif dimension == '3D':
		projection_shortcut = convolution_3D
	else:
		raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

	with tf.variable_scope(name):
		# The projection_shortcut should come after the batch norm and ReLU.
		inputs = batch_norm(inputs, training, 'batch_norm')
		inputs = relu(inputs, 'relu')
		shortcut = projection_shortcut(inputs, output_filters, 3, 2, False, 'projection_shortcut')
		inputs, _ = self_attention(
						inputs,
						output_filters // conf_attn_down['key_ratio'],
						output_filters // conf_attn_down['value_ratio'],
						output_filters,
						conf_attn_down['num_heads'],
						training,
						dimension,
						'DOWN',
						'attention',
						conf_attn_down['dropout_rate'],
						conf_attn_down['use_softmax'],
						conf_attn_down['use_bias'])
		return tf.add(shortcut, inputs)


def up_gto_v2(inputs, output_filters, training, dimension, name):
	"""Up GTO block version 2. (Yaochen)"""
	if conf_attn_up['key_ratio'] != 1:
		raise ValueError("Must set key_ratio == 1!")
		
	with tf.variable_scope(name):
		inputs = batch_norm(inputs, training, 'batch_norm')
		inputs = relu(inputs, 'relu')
		inputs, query = self_attention(
							inputs,
							output_filters // conf_attn_up['key_ratio'],
							output_filters // conf_attn_up['value_ratio'],
							output_filters,
							conf_attn_up['num_heads'],
							training,
							dimension,
							'UP',
							'attention',
							conf_attn_up['dropout_rate'],
							conf_attn_up['use_softmax'],
							conf_attn_up['use_bias'])
		return tf.add(query, inputs)


def up4_gto_v2(inputs, output_filters, training, dimension, name):
	"""4 times upsampling, used for projection models, e.g. Flywing Projection"""
	if conf_attn_up['key_ratio'] != 1:
		raise ValueError("Must set key_ratio == 1!")

	with tf.variable_scope(name):
		inputs = batch_norm(inputs, training, 'batch_norm')
		inputs = relu(inputs, 'relu')
		inputs, query = self_attention(
							inputs,
							output_filters // conf_attn_up['key_ratio'],
							output_filters // conf_attn_up['value_ratio'],
							output_filters,
							conf_attn_up['num_heads'],
							training,
							dimension,
							'UP4',
							'attention',
							conf_attn_up['dropout_rate'],
							conf_attn_up['use_softmax'],
							conf_attn_up['use_bias'])
		return tf.add(query, inputs)


def down_gto_v2(inputs, output_filters, training, dimension, name):
	"""Down GTO block version 2. (Yaochen)"""
	if conf_attn_down['key_ratio'] != 1:
		raise ValueError("Must set key_ratio == 1!")

	with tf.variable_scope(name):
		inputs = batch_norm(inputs, training, 'batch_norm')
		inputs = relu(inputs, 'relu')
		inputs, query = self_attention(
							inputs,
							output_filters // conf_attn_down['key_ratio'],
							output_filters // conf_attn_down['value_ratio'],
							output_filters,
							conf_attn_down['num_heads'],
							training,
							dimension,
							'DOWN',
							'attention',
							conf_attn_down['dropout_rate'],
							conf_attn_down['use_softmax'],
							conf_attn_down['use_bias'])
		return tf.add(query, inputs)
