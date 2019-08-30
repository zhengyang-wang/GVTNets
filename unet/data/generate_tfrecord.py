import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import transforms
import tifffile

from czidataset import CziDataset
from prepare_test import prepare_test
from tqdm import tqdm

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

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
	# print(dataset)
	return dataset

def generate_tfrecord(opts, test):
	# .csv -> .tfrecords
	dataset = get_dataset(opts, test)
	
	if not test:
		output_file = os.path.join(opts.tf_dataset_dir, 'train.tfrecords')
		if os.path.isfile(output_file):
			print('The tfrecord file already exists.')
			return 0

		writer = tf.python_io.TFRecordWriter(output_file)

		pbar = tqdm(range(0, opts.num_train_pairs))
		for i in pbar:
			pbar.set_description("generating tfrecords")
			data = dataset[i]

			if len(data) == 1: # no target
				data.append(data[0]) # use source as pseudo target

			example = tf.train.Example(features=tf.train.Features(
				feature={
					'source': _bytes_feature([data[0].tostring()]),
					'target': _bytes_feature([data[1].tostring()]),
					'shape': _int64_feature(data[0].shape),
				}
			))
			writer.write(example.SerializeToString())

		writer.close()
		print('The tfrecord file has been created: %s' % (output_file))

	else:
		output_dir = os.path.join(opts.tf_dataset_dir, 'test')
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		output_dir = os.path.join(output_dir, 'patch%s_step%s' % (str(opts.test_patch_size), str(opts.test_step_size)))
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		else:
			print('The tfrecord files already exist.')
			return 0

		for i in range(0, opts.num_test_pairs):
			name = dataset.get_information(i)['path_czi']
			print('Processing %s' % name)

			data = dataset[i]

			if not os.path.exists(opts.result_dir):
				os.makedirs(opts.result_dir)
			path_tiff = os.path.join(opts.result_dir, 'signal_{:02d}.tiff'.format(i))
			if not os.path.isfile(path_tiff):
				tifffile.imsave(path_tiff, data[0])
				print('saved:', path_tiff)

			if len(data) == 2 and not opts.no_target:
				if not os.path.exists(opts.result_dir):
					os.makedirs(opts.result_dir)
				path_tiff = os.path.join(opts.result_dir, 'target_{:02d}.tiff'.format(i))
				if not os.path.isfile(path_tiff):
					tifffile.imsave(path_tiff, data[1])
					print('saved:', path_tiff)

			if len(data) == 1: # no target
				data.append(data[0]) # use source as pseudo target

			patch_ids, real_patch_size = prepare_test(data[0], opts.test_patch_size, opts.test_step_size)

			output_file = os.path.join(output_dir, 'test_%d.tfrecords' % i)
			writer = tf.python_io.TFRecordWriter(output_file)

			for j in range(len(patch_ids)):
				(d, h, w) = patch_ids[j]

				source = data[0][d:d+real_patch_size[0], h:h+real_patch_size[1], w:w+real_patch_size[2]]

				example = tf.train.Example(features=tf.train.Features(
					feature={
						'source': _bytes_feature([source.tostring()]),
						'patch_size': _int64_feature(real_patch_size),
					}
				))
				writer.write(example.SerializeToString())

			writer.close()
			print('The tfrecord file has been created: %s' % (output_file))

def main():
	parser = argparse.ArgumentParser()
	factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
	default_resizer_str = 'transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
	parser.add_argument('--csv_dataset_dir', type=str, help='directory of csv files for constructing Datasets')
	parser.add_argument('--raw_dataset_dir', type=str, help='directory of raw data files')
	parser.add_argument('--tf_dataset_dir', type=str, help='directory of tfrecord files')
	parser.add_argument('--num_train_pairs', type=int, default=None, help='number of pairs for training')
	parser.add_argument('--num_test_pairs', type=int, default=None, help='number of pairs for testing')
	parser.add_argument('--test_patch_size', nargs='+', type=int, default=[0, 0, 0], help='size of patches to cut Dataset elements during testing.\
							zero means the patch size is set to the data size')
	parser.add_argument('--test_step_size', nargs='+', type=int, default=[0, 0, 0], help='size of step between patches during testing.\
							zero means the step size is set to the patch size')
	parser.add_argument('--result_dir', type=str, help='directory of resulted tiff files')
	parser.add_argument('--no_target', action='store_true', help='set to not save target image')
	parser.add_argument('--transform_signal', nargs='+', default=['transforms.normalize', default_resizer_str], help='list of transforms on Dataset signal')
	parser.add_argument('--transform_target', nargs='+', default=['transforms.normalize', default_resizer_str], help='list of transforms on Dataset target')
	opts = parser.parse_args()

	if not os.path.exists(opts.tf_dataset_dir):
		os.makedirs(opts.tf_dataset_dir)

	if opts.num_train_pairs != None:
		generate_tfrecord(opts, test=False)

	if opts.num_test_pairs != None:
		generate_tfrecord(opts, test=True)

if __name__ == '__main__':
	main()