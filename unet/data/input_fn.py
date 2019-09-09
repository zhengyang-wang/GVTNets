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
		for i in range(opts.num_iters):
			if shuffle:
				idx = np.random.randint(len(sources))
			else:
				idx = i // len(sources)
			source, target = sources[idx], targets[idx]

			# random crop
			# TODO: compatability to 2D images
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


def input_fn_tfrecord(opts, fnames):

	pass



def save_tfrecord(opts, source, target):

	pass





