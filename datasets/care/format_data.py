import tifffile, os
import numpy as np
import argparse

def format_data(raw_file_path):
	image_list = os.listdir(raw_file_path)
	for condition in ['GT', 'condition_1', 'condition_2', 'condition_3']:
		output_dir = os.path.join(raw_file_path, condition)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		else:
			print('The training dataset already exists.')
			return 0

	for fname in image_list:
		img_dir = os.path.join(raw_file_path, fname)
		img = tifffile.imread(img_dir)
		gt = img[3]
		c1 = img[2]
		c2 = img[1]
		c3 = img[0]
		tifffile.imsave(os.path.join(raw_file_path, 'GT', fname), gt)
		tifffile.imsave(os.path.join(raw_file_path, 'condition_1', fname), c1)
		tifffile.imsave(os.path.join(raw_file_path, 'condition_2', fname), c2)
		tifffile.imsave(os.path.join(raw_file_path, 'condition_3', fname), c3)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--raw_test_dataset_dir', type=str, help='directory of test files for datasets')
	opts = parser.parse_args()

	format_data(opts.raw_test_dataset_dir)