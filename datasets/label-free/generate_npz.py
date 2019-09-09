import os
import sys
import argparse
import tifffile
import numpy as np

import transforms
from czidataset import CziDataset


def get_dataset(opts, test):
	transform_signal = [eval(t) for t in opts.transform_signal]
	transform_target = [eval(t) for t in opts.transform_target]

	if test:
		propper = transforms.Propper()
		# print(propper)
		transform_signal.append(propper)
		transform_target.append(propper)

	dataset = CziDataset(
		raw_dataset_dir = opts.raw_dataset_dir,
		path_csv = os.path.join(opts.csv_dataset_dir, 'train.csv') if not test else os.path.join(opts.csv_dataset_dir, 'test.csv'),
		transform_source = transform_signal,
		transform_target = transform_target,
	)
	return dataset


def generate_files(opts, test):
	# Save training datasets into the npz format.
	# https://kite.com/python/docs/numpy.lib.npyio.NpzFile
	# https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
	# Save testing datasets into tiff images.
	dataset = get_dataset(opts, test)

	if not test:
		# Training data will be saved in npz files.
		# FORMAT: {'X': (n_sample, n_channel, depth, height, width),
		#		   'Y': (n_sample, n_channel, depth, height, width)}
		output_dir = os.path.join(opts.npz_dataset_dir, 'train')
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		else:
			print('The training dataset already exists.')
			return 0

		for i in range(0, opts.num_train_pairs):
			# Here, as training data have different sizes, each training sample
			# will be saved into a single npz file.
			output_file = os.path.join(output_dir, 'train_{:02d}.npz'.format(i))
			data = dataset[i]

			# Add the channel dimension.
			X = np.expand_dims(data[0], axis=0).astype(np.float32)
			Y = np.expand_dims(data[1], axis=0).astype(np.float32)

			np.savez(output_file, X=X, Y=Y)
			print('The training npz file has been created: %s' % (output_file))
	else:
		# Each image for prediction is saved as signal_{index}.tiff
		# The corresponding label, if present, is saved as target_{index}.tiff
		output_dir = os.path.join(opts.npz_dataset_dir, 'test')
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		else:
			print('The testing dataset already exists.')
			return 0

		source_output_dir = os.path.join(output_dir, 'sources')
		if not os.path.exists(source_output_dir):
			os.makedirs(source_output_dir)

		label_output_dir = os.path.join(output_dir, 'targets')
		if not os.path.exists(label_output_dir):
			os.makedirs(label_output_dir)

		for i in range(0, opts.num_test_pairs):
			name = dataset.get_information(i)['path_czi']
			print('Processing %s' % name)

			data = dataset[i]

			path_tiff = os.path.join(source_output_dir, 'source_{:02d}.tiff'.format(i))
			if not os.path.isfile(path_tiff):
				tifffile.imsave(path_tiff, data[0])
				print('Saved:', path_tiff)

			if len(data) == 2 and not opts.no_target:
				path_tiff = os.path.join(label_output_dir, 'target_{:02d}.tiff'.format(i))
				if not os.path.isfile(path_tiff):
					tifffile.imsave(path_tiff, data[1])
					print('Saved:', path_tiff)


def main():
	parser = argparse.ArgumentParser()
	factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
	default_resizer_str = 'transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
	parser.add_argument('--csv_dataset_dir', type=str, help='directory of csv files for constructing datasets')
	parser.add_argument('--raw_dataset_dir', type=str, help='directory of raw data files')
	parser.add_argument('--npz_dataset_dir', type=str, help='directory of npz files')
	parser.add_argument('--num_train_pairs', type=int, default=None, help='number of pairs for training')
	parser.add_argument('--num_test_pairs', type=int, default=None, help='number of pairs for testing')
	parser.add_argument('--no_target', action='store_true', help='set to not save target image')
	parser.add_argument('--transform_signal', nargs='+', default=['transforms.normalize', default_resizer_str], help='list of transforms on Dataset signal')
	parser.add_argument('--transform_target', nargs='+', default=['transforms.normalize', default_resizer_str], help='list of transforms on Dataset target')
	opts = parser.parse_args()

	if not os.path.exists(opts.npz_dataset_dir):
		os.makedirs(opts.npz_dataset_dir)

	if opts.num_train_pairs != None:
		generate_files(opts, test=False)

	if opts.num_test_pairs != None:
		generate_files(opts, test=True)

if __name__ == '__main__':
	main()