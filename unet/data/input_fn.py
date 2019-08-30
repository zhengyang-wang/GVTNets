import tensorflow as tf
import os
import numpy as np

def input_function_npz(opts, mode):
	if mode == 'train':
		data = np.load(opts.npz_dataset_dir+'/data_label.npz')
		source = np.moveaxis(data['X'], 1, -1)
		target = np.moveaxis(data['Y'], 1, -1)

		repeats = opts.num_iters * opts.batch_size // opts.num_train_pairs
		input_fn = tf.estimator.inputs.numpy_input_fn(
						x=source, y=target, batch_size=opts.batch_size,
						num_epochs=repeats, shuffle=True)
	elif mode == 'pred':
		data = np.load(opts.npz_dataset_dir+'/test_data.npz')
		source = np.moveaxis(data['X'], 1, -1)
		input_fn = tf.estimator.inputs.numpy_input_fn(
						x=source, y=None, batch_size=1,
						num_epochs=1, shuffle=True)
	return input_fn()

def get_filenames(opts, mode):
	if mode == 'train':
		return [os.path.join(opts.tf_dataset_dir, 'train.tfrecords')]
	else:
		dataset_dir = os.path.join(opts.tf_dataset_dir, 'test', 'patch%s_step%s' % (str(opts.test_patch_size), str(opts.test_step_size)))
		return [os.path.join(dataset_dir, 'test_%d.tfrecords' % i) for i in range(0, opts.num_test_pairs)]

def decode_train(serialized_example):
	"""Parses training data from the given `serialized_example`."""
	features = tf.parse_single_example(
					serialized_example,
					features={
						'source':tf.FixedLenFeature([],tf.string),
						'target':tf.FixedLenFeature([], tf.string),
						'shape':tf.FixedLenFeature(3, tf.int64),
					})

	# Convert from a scalar string tensor
	source = tf.decode_raw(features['source'], tf.float32)
	source = tf.reshape(source, features['shape'])
	target = tf.decode_raw(features['target'], tf.float32)
	target = tf.reshape(target, features['shape'])

	return source, target

def decode_test(serialized_example):
	"""Parses testing data from the given `serialized_example`."""
	features = tf.parse_single_example(
					serialized_example,
					features={
						'source':tf.FixedLenFeature([],tf.string),
						'patch_size':tf.FixedLenFeature(3, tf.int64),
					})

	# Convert from a scalar string tensor
	source = tf.decode_raw(features['source'], tf.float32)
	source = tf.reshape(source, features['patch_size'])
	target = tf.constant(0) # pseudo target

	return source, target

def crop_image_train(source, target, patch_size):
	"""Crop training data."""
	pair = tf.stack([source, target], axis=-1)
	pair = tf.random_crop(pair, patch_size+[2])
	[source, target] = tf.unstack(pair, 2, axis=-1)

	return source, target

def add_channel_dim(source, target):
	"""Add an additional (last) dimension as the channel."""
	source = tf.expand_dims(source, axis=-1)
	target = tf.expand_dims(target, axis=-1)

	return source, target

def input_function(opts, mode):
	"""Input function.

	Returns:
		Dataset of (features, labels) pairs ready for iteration.
	"""

	with tf.name_scope('input'):
		# Generate a Dataset with raw records.
		filenames = get_filenames(opts, mode)
		dataset = tf.data.TFRecordDataset(filenames)

		# We prefetch a batch at a time, This can help smooth out the time taken to
		# load input files as we go through shuffling and processing.
		dataset = dataset.prefetch(buffer_size=opts.batch_size)

		if mode == 'train':
			# Shuffle the records. Note that we shuffle before repeating to ensure
			# that the shuffling respects epoch boundaries.
			dataset = dataset.shuffle(buffer_size=opts.buffer_size)

			# If we are training over multiple epochs before evaluating, repeat the
			# dataset for the appropriate number of epochs.
			dataset = dataset.repeat(opts.num_iters * opts.batch_size // opts.num_train_pairs)
			dataset = dataset.map(decode_train, num_parallel_calls=opts.num_parallel_calls)
			dataset = dataset.map(lambda x, y: crop_image_train(x, y, opts.patch_size), num_parallel_calls=opts.num_parallel_calls)

		elif mode == 'pred':
			dataset = dataset.repeat(1)
			dataset = dataset.map(decode_test, num_parallel_calls=opts.num_parallel_calls)
		
		dataset = dataset.map(add_channel_dim, num_parallel_calls=opts.num_parallel_calls)

		dataset = dataset.batch(opts.batch_size)

		# Operations between the final prefetch and the get_next call to the iterator
		# will happen synchronously during run time. We prefetch here again to
		# background all of the above processing work and keep it out of the
		# critical training path.
		dataset = dataset.prefetch(1)

		iterator = dataset.make_one_shot_iterator()
		features, labels = iterator.get_next()

		return features, labels
