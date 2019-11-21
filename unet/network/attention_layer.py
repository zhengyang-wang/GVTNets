import tensorflow as tf
from .basic_ops import *


"""This script defines 2D and 3D multihead self-attention layers.
"""


def self_attention(inputs, total_key_filters, total_value_filters, output_filters,
		num_heads, training, dimension, layer_type, name, dropout_rate=0.0, use_softmax=True,
		use_bias=True):
	"""Multihead scaled-dot-product attention with input/output transformations.
	
	Args:
		inputs: a Tensor with shape [batch, (d,) h, w, channels]
		total_key_filters: an integer. Note that queries have the same number 
			of channels as keys.
		total_value_filters: an integer
		output_filters: an integer
		num_heads: an integer dividing total_key_filters and total_value_filters
		training: a boolean for dropout
		dimension: a string, dimension of inputs/outputs -- 2D, 3D
		layer_type: a string, type of this layer -- SAME, DOWN, UP, UP4
		name: a string
		dropout_rate: a float between 0.0 and 1.0. No dropout if dropout_rate = 0.0
		use_softmax: a boolean deciding whether to use softmax. Note that use_softmax = False
			will automatically set dropout_rate = 0.0
		use_bias: a boolean deciding whether to use the bias term in input/output transformations

	Returns:
		A Tensor of shape [batch, (_d,) _h, _w, output_filters]
	
	Raises:
		ValueError: if the total_key_filters or total_value_filters are not divisible
			by the number of attention heads.
		ValueError: if dimension is not one of ['2D', '3D'].
		ValueError: if layer_type is not one of ['SAME', 'DOWN', 'UP', 'UP4'].
	"""
	if total_key_filters % num_heads != 0:
		raise ValueError("Key depth (%d) must be divisible by the number of "
						"attention heads (%d)." % (total_key_filters, num_heads))
	if total_value_filters % num_heads != 0:
		raise ValueError("Value depth (%d) must be divisible by the number of "
						"attention heads (%d)." % (total_value_filters, num_heads))
	if layer_type not in ['SAME', 'DOWN', 'UP', 'UP4']:
		raise ValueError("Layer type (%s) must be one of SAME, DOWN, UP, UP4." % (layer_type))

	if dimension == '2D':
		compute_qkv = compute_qkv_2d
		split_heads = split_heads_2d
		unfold = unfold_2d
		combine_heads = combine_heads_2d
		output_transfrom = convolution_2D
	elif dimension == '3D':
		compute_qkv = compute_qkv_3d
		split_heads = split_heads_3d
		unfold = unfold_3d
		combine_heads = combine_heads_3d
		output_transfrom = convolution_3D
	else:
		raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

	with tf.variable_scope(name):
		# produce q, k, v
		q, k, v = compute_qkv(inputs, total_key_filters, total_value_filters, use_bias,
					layer_type)

		# after splitting, shape is [batch, heads, d, h, w, channels / heads]
		q_split = split_heads(q, num_heads)
		k_split = split_heads(k, num_heads)
		v_split = split_heads(v, num_heads)

		# normalization recommended by "Attention is All You Need"
		key_filters_per_head = total_key_filters // num_heads
		q_split *= key_filters_per_head**-0.5

		output_shape = tf.concat([tf.shape(q_split)[0:-1], [v_split.shape[-1].value]], 0)

		# flatten q,k,v
		q_new = unfold(q_split)
		k_new = unfold(k_split)
		v_new = unfold(v_split)

		# attention
		o = dot_product_attention(q_new, k_new, v_new, training, dropout_rate, use_softmax)

		# putting the representations back in the right place
		o = tf.reshape(o, output_shape)

		# combine heads and perform output transformation
		o = combine_heads(o)

		o = output_transfrom(o, output_filters, 1, 1, use_bias, 'out_transform')

		return o, q


def compute_qkv_2d(inputs, total_key_filters, total_value_filters, use_bias, layer_type):
	"""Perform qkv transformations and compute query, key and value.

	Args:
		inputs: a Tensor with shape [batch, h, w, channels]
		total_key_filters: an integer
		total_value_filters: an integer
		use_bias: a boolean deciding whether to use the bias term in qkv transformations
		layer_type: a string, type of this layer -- SAME, DOWN, UP
	
	Returns:
		q: a Tensor with shape [batch, _h, _w, total_key_filters]
		k: a Tensor with shape [batch, h, w, total_key_filters]
		v: a Tensor with shape [batch, h, w, total_value_filters]
	"""
	# transformation for q
	if layer_type == 'SAME':
		q = convolution_2D(inputs, total_key_filters, 1, 1, use_bias, 'q_transform')
	elif layer_type == 'DOWN':
		q = convolution_2D(inputs, total_key_filters, 3, 2, use_bias, 'q_transform')
	elif layer_type == 'UP':
		q = transposed_convolution_2D(inputs, total_key_filters, 3, 2, use_bias, 'q_transform')

	# linear transformation for k
	k = convolution_2D(inputs, total_key_filters, 1, 1, use_bias, 'k_transform')

	# linear transformation for v
	v = convolution_2D(inputs, total_value_filters, 1, 1, use_bias, 'v_transform')

	return q, k, v


def compute_qkv_3d(inputs, total_key_filters, total_value_filters, use_bias, layer_type):
	"""Perform qkv transformations and compute query, key and value.

	Args:
		inputs: a Tensor with shape [batch, d, h, w, channels]
		total_key_filters: an integer
		total_value_filters: an integer
		use_bias: a boolean deciding whether to use the bias term in qkv transformations
		layer_type: a string, type of this layer -- SAME, DOWN, UP
	
	Returns:
		q: a Tensor with shape [batch, _d, _h, _w, total_key_filters]
		k: a Tensor with shape [batch, d, h, w, total_key_filters]
		v: a Tensor with shape [batch, d, h, w, total_value_filters]
	"""
	# transformation for q
	if layer_type == 'SAME':
		q = convolution_3D(inputs, total_key_filters, 1, 1, use_bias, 'q_transform')
	elif layer_type == 'DOWN':
		q = convolution_3D(inputs, total_key_filters, 3, 2, use_bias, 'q_transform')
	elif layer_type == 'UP':
		q = transposed_convolution_3D(inputs, total_key_filters, 3, 2, use_bias, 'q_transform')
	# ProjectionNet uses 4 times up-sampling and down-sampling. For projection models only, e.g. Flywing Projections.
	elif layer_type == 'UP4':
		q = tf.reshape(inputs, tf.concat([tf.shape(inputs)[0:1]*tf.shape(inputs)[1:2], tf.shape(inputs)[2:]],0))
		q = tf.image.resize_nearest_neighbor(q, tf.concat([tf.shape(inputs)[2:3]*4, tf.shape(inputs)[3:4]*4],0))
		q = tf.reshape(q, tf.concat([tf.shape(inputs)[:2], tf.shape(q)[1:]], 0))
		
	# linear transformation for k
	k = convolution_3D(inputs, total_key_filters, 1, 1, use_bias, 'k_transform')

	# linear transformation for v
	v = convolution_3D(inputs, total_value_filters, 1, 1, use_bias, 'v_transform')

	return q, k, v


def reshape_range(tensor, i, j, shape):
	"""Reshapes a tensor between dimensions [i,j)."""
	target_shape = tf.concat(
			[tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
			axis=0)
	return tf.reshape(tensor, target_shape)


def unfold_2d(x):
	x_shape = tf.shape(x)
	# [batch, heads, length, channels], length = h*w
	x = reshape_range(x, 2, 4, [tf.reduce_prod(x_shape[2:4])])
	return x


def unfold_3d(x):
	x_shape = tf.shape(x)
	# [batch, heads, length, channels], length = d*h*w
	x = reshape_range(x, 2, 5, [tf.reduce_prod(x_shape[2:5])])
	return x


def dot_product_attention(q, k, v, training, dropout_rate, use_softmax):
	"""Dot-product attention.

	Args:
		q: a Tensor with shape [batch, heads, length_q, channels_k]
		k: a Tensor with shape [batch, heads, length_kv, channels_k]
		v: a Tensor with shape [batch, heads, length_kv, channels_v]
		training: a boolean for dropout
		dropout_rate: a float between 0.0 and 1.0. No dropout if dropout_rate = 0.0
		use_softmax: a boolean deciding whether to use softmax. Note that
			use_softmax = False will automatically set dropout_rate = 0.0

	Returns:
		A Tensor with shape [batch, heads, length_q, channels_v]
	"""
	if use_softmax:
		# [batch, num_heads, length_q, length_kv]
		attention_weights = tf.matmul(q, k, transpose_b=True)

		# normalize attention
		attention_weights = tf.nn.softmax(attention_weights)

		# dropping out the attention links for each of the heads
		if dropout_rate != 0.0:
			attention_weights = tf.layers.dropout(attention_weights, dropout_rate, training)

		return tf.matmul(attention_weights, v)
	else:
		# To save computation, compute the multiplication between K^T and V first.
		kv = tf.matmul(k, v, transpose_a=True)

		# normalize
		kv = kv/tf.cast(tf.shape(q)[2], tf.float32)

		return tf.matmul(q, kv)


def split_heads_2d(x, num_heads):
	"""Split channels (last dimension) into multiple heads (becomes dimension 1).
	
	Args:
		x: a Tensor with shape [batch, h, w, channels]
		num_heads: an integer
	
	Returns:
		a Tensor with shape [batch, num_heads, h, w, channels / num_heads]
	"""
	return tf.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def split_heads_3d(x, num_heads):
	"""Split channels (last dimension) into multiple heads (becomes dimension 1).
	
	Args:
		x: a Tensor with shape [batch, d, h, w, channels]
		num_heads: an integer
	
	Returns:
		a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]
	"""
	return tf.transpose(split_last_dimension(x, num_heads), [0, 4, 1, 2, 3, 5])


def split_last_dimension(x, n):
	"""Reshape x so that the last dimension becomes two dimensions.
	The first of these two dimensions is n.

	Args:
		x: a Tensor with shape [..., m]
		n: an integer.

	Returns:
		a Tensor with shape [..., n, m/n]
	"""
	old_shape = x.get_shape().dims
	last = old_shape[-1]
	new_shape = old_shape[:-1] + [n] + [last // n if last else None]
	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
	ret.set_shape(new_shape)
	return ret


def combine_heads_2d(x):
	"""Inverse of split_heads_2d.

	Args:
		x: a Tensor with shape [batch, num_heads, h, w, channels / num_heads]

	Returns:
		a Tensor with shape [batch, h, w, channels]
	"""
	return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 1, 4]))


def combine_heads_3d(x):
	"""Inverse of split_heads_3d.

	Args:
		x: a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]

	Returns:
		a Tensor with shape [batch, d, h, w, channels]
	"""
	return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 4, 1, 5]))


def combine_last_two_dimensions(x):
	"""Reshape x so that the last two dimension become one.

	Args:
		x: a Tensor with shape [..., a, b]

	Returns:
		a Tensor with shape [..., a*b]
	"""
	old_shape = x.get_shape().dims
	a, b = old_shape[-2:]
	new_shape = old_shape[:-2] + [a * b if a and b else None]
	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
	ret.set_shape(new_shape)
	return ret
