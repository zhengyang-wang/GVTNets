import tensorflow as tf
import os
import numpy as np

def load_npz(opts):
	""" Loading data from npz files. 
	    Source image(s) and target image(s) are saved as 'X' and 'Y' per npz file with dtype 'float32'.
	    Could be a single npz file of shapes [n_samples, n_channels, (D,) W, H],
	    or a list of files of shape [n_channels, (D,) W, H] (for images with different sizes).
	"""
	fnames = [fname for fname in os.listdir(opts.npz_dataset_dir) if 
			fnmatch.fnmatch(file, '*.npz')]
	if len(fnames)==1:
		data = np.load(opts.npz_dataset_dir)
		# moving channel dimension to the last
		sources = np.moveaxis(data['X'], 1, -1)
		targets = np.moveaxis(data['Y'], 1, -1)
	else:
		sources = []
		targets = []
		for fname in fnames:
			data = np.load(opts.npz_dataset_dir)
			sources.append(np.moveaxis(data['X'], 0, -1))
			targets.append(np.moveaxis(data['Y'], 0, -1))
		
	return sources, targets


def input_function(opts, mode):
	'''
	Args:
		source: source images of shape [n_samples, (D,) H, W, n_channels]
		target: target images of shape [n_samples, (D,) H, W, n_channels]
	'''
	# TODO: implement for prediction input
	sources, targets = load_npz(opts)
	if opts.cropped:
		input_fn = input_fn_numpy(source, target, opts)
	elif not opts.save_tfrecords:
		input_fn = input_fn_generator(source, target, opts)
	else:
		save_tfrecord(opts, source, target)
		input_fn = input_fn_tfrecord(opts)

	return input_fn


def input_fn_numpy(source, target, opts, shuffle=True, seed=None):

	repeats = opts.num_iters * opts.batch_size // opts.num_train_pairs

	return tf.estimator.inputs.numpy_input_fn(
				x=source, y=target, batch_size=opts.batch_size,
				num_epochs=repeats, shuffle=shuffle)


def input_fn_generator(source, target, opts, shuffle=True, seed=None):

	def generator():
		while(True):
			idx = np.random.randint(source.shape[0])
			src,trg = source[idx], target[idx]
			valid_shape = src.shape[:-1]-np.array(opts.patch_size)
			# TODO: compatability to 2D images
			z = np.random.randint(0,valid_shape[0])
			x = np.random.randint(0,valid_shape[1])
			y = np.random.randint(0,valid_shape[2])
			s = (slice(z,z+patch_size[0]), 
				 slice(x,x+patch_size[1]), 
				 slice(y,y+patch_size[2]))
			src_ptch = src[s]
			trg_ptch = trg[s]
			yield src_ptch, trg_ptch

	output_types = (tf.float32, tf.float32)
	output_shapes = (tf.TensorShape(patch_size+(1,)), 
					 tf.TensorShape(patch_size+(1,)))
	dataset = tf.data.Dataset.from_generator(generator, 
		output_types=output_types, output_shapes=output_shapes)
	dataset = dataset.batch(batch_size).prefetch(4)

	return dataset


def input_fn_tfrecord(opts, fnames):

	pass



def save_tfrecord(opts, source, target):

	pass





