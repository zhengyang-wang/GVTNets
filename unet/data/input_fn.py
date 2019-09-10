import tensorflow as tf
import os
import numpy as np
import fnmatch


def load_training_npz(npz_dataset_dir, num_train_pairs):
	""" Loading data from npz files.
	DTYPE: np.float32
	FORMAT options:
		A single npz file containing all training data:
			{'X': (n_sample, n_channel, (depth,) height, width),
			 'Y': (n_sample, n_channel, (depth,) height, width)}
		Multiple npz files where each one contains one training sample:
			NOTE: (depth,) height, width) can vary for different samples.
			{'X': (n_channel, (depth,) height, width),
			 'Y': (n_channel, (depth,) height, width)}

	Return:
		A single npz file containing all training data:
			sources: An numpy array of shape [n_sample, (depth,) height, width, n_channel]
			targets: An numpy array of shape [n_sample, (depth,) height, width, n_channel]
		Multiple npz files where each one contains one training sample:
			NOTE: (depth,) height, width) can vary for different samples.
			sources: A list of numpy arrays of shape [(depth,) height, width, n_channel]
			targets: A list of numpy arrays of shape [(depth,) height, width, n_channel]
	"""
	print("Loading npz file(s)...")
	fnames = [fname for fname in os.listdir(npz_dataset_dir) if 
				fnmatch.fnmatch(fname, '*.npz')]
	fnames.sort()
	fnames = [os.path.join(npz_dataset_dir, fname) for fname in fnames]

	if len(fnames)==1:
		data = np.load(fnames[0])
		# moving channel dimension to the last
		sources = np.moveaxis(data['X'], 1, -1)
		targets = np.moveaxis(data['Y'], 1, -1)

	else:
		sources = []
		targets = []

		for fname in fnames:
			data = np.load(fname)
			sources.append(np.moveaxis(data['X'], 0, -1))
			targets.append(np.moveaxis(data['Y'], 0, -1))
		
		assert len(sources) == num_train_pairs, "len(sources) is %d" % len(sources)

	print("Data loaded.")
	return sources, targets


def input_function(opts, mode):
	# TODO: implement for prediction input
	sources, targets = load_training_npz(opts.npz_dataset_dir, opts.num_train_pairs)

	if opts.already_cropped:
		# The training data have been cropped.
		# The training data are stored in a single npz file.
		input_fn = input_fn_numpy(sources, targets, opts)

	elif not opts.save_tfrecords:
		# The training data have NOT been cropped.
		# The training data are stored in multiple npz files,
		# where each one contains one training sample.
		input_fn = lambda: input_fn_generator(sources, targets, opts)

	else:
		# Using tfrecords can handle both cases.
		save_tfrecord(opts, sources, targets)
		input_fn = lambda: input_fn_tfrecord(opts)

	return input_fn


def input_fn_numpy(sources, targets, opts, shuffle=True):
	repeats = opts.num_iters * opts.batch_size // opts.num_train_pairs

	return tf.estimator.inputs.numpy_input_fn(
				x=sources, y=targets, batch_size=opts.batch_size,
				num_epochs=repeats, shuffle=shuffle)


def input_fn_generator(sources, targets, opts, shuffle=True):

	def generator():
		for i in range(opts.num_iters * opts.batch_size):
			if shuffle:
				idx = np.random.randint(len(sources))
			else:
				idx = i // len(sources)
			source, target = sources[idx], targets[idx]

			# random crop
			# TODO: compatability to 2D inputs
			valid_shape = source.shape[:-1] - np.array(opts.train_patch_size)
			z = np.random.randint(0, valid_shape[0])
			x = np.random.randint(0, valid_shape[1])
			y = np.random.randint(0, valid_shape[2])
			s = (slice(z, z+opts.train_patch_size[0]), 
				 slice(x, x+opts.train_patch_size[1]), 
				 slice(y, y+opts.train_patch_size[2]))
			source_patch = source[s]
			target_patch = target[s]

			yield source_patch, target_patch

	output_types = (tf.float32, tf.float32)
	output_shapes = (tf.TensorShape(opts.train_patch_size + [1]), 
					 tf.TensorShape(opts.train_patch_size + [1]))
	dataset = tf.data.Dataset.from_generator(generator, 
					output_types=output_types, output_shapes=output_shapes)
	dataset = dataset.batch(opts.batch_size).prefetch(1)

	return dataset


def input_fn_tfrecord(opts):

	# TODO: compatability to 2D inputs
	def decode_train(serialized_example):
		"""Parses training data from the given `serialized_example`."""
		features = tf.parse_single_example(
						serialized_example,
						features={
							'source':tf.FixedLenFeature([],tf.string),
							'target':tf.FixedLenFeature([], tf.string),
							'shape':tf.FixedLenFeature(4, tf.int64),
						})

		# Convert from a scalar string tensor
		source = tf.decode_raw(features['source'], tf.float32)
		source = tf.reshape(source, features['shape'])
		target = tf.decode_raw(features['target'], tf.float32)
		target = tf.reshape(target, features['shape'])
		return source, target

	def crop_image_train(source, target, patch_size):
		"""Crop training data."""
		pair = tf.concat([source, target], axis=-1)
		pair = tf.random_crop(pair, patch_size+[2])
		[source, target] = tf.split(pair, 2, axis=-1)
		return source, target

	dataset = tf.data.TFRecordDataset([os.path.join(opts.tf_dataset_dir, 'train.tfrecords')])
	# We prefetch a batch at a time, This can help smooth out the time taken to
	# load input files as we go through shuffling and processing.
	dataset = dataset.prefetch(buffer_size=opts.batch_size)
	# Shuffle the records. Note that we shuffle before repeating to ensure
	# that the shuffling respects epoch boundaries.
	dataset = dataset.shuffle(buffer_size=30)
	# If we are training over multiple epochs before evaluating, repeat the
	# dataset for the appropriate number of epochs.
	dataset = dataset.repeat(opts.num_iters * opts.batch_size // opts.num_train_pairs)
	dataset = dataset.map(decode_train, num_parallel_calls=5)
	dataset = dataset.map(lambda x, y: crop_image_train(x, y, opts.patch_size), num_parallel_calls=5)
	dataset = dataset.batch(opts.batch_size)
	# Operations between the final prefetch and the get_next call to the iterator
	# will happen synchronously during run time. We prefetch here again to
	# background all of the above processing work and keep it out of the
	# critical training path.
	dataset = dataset.prefetch(1)

	return dataset


def save_tfrecord(sources, targets, opts):

	def _bytes_feature(value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

	def _int64_feature(value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

	if not os.path.exists(opts.tf_dataset_dir):
		os.makedirs(opts.tf_dataset_dir)
	else:
		print('The tf_dataset_dir already exists.')
		return 0

	output_file = os.path.join(opts.tf_dataset_dir, 'train.tfrecords')

	writer = tf.python_io.TFRecordWriter(output_file)

	print("Creating train.tfrecords...")
	for i in range(len(sources)):
		source, target = sources[i], targets[i]

		example = tf.train.Example(features=tf.train.Features(
			feature={
				'source': _bytes_feature([source.tostring()]),
				'target': _bytes_feature([target.tostring()]),
				'shape': _int64_feature(source.shape),
			}
		))
		writer.write(example.SerializeToString())

	writer.close()
	print('The train.tfrecords has been created: %s' % (output_file))
