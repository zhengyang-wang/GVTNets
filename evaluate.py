import os
import argparse

from unet.data import evaluate_function

def main():
	parser = argparse.ArgumentParser()
	# You need to specific either prediction_file + target_file or prediction_dir + target_dir + stats_file
	# evaluate one file
	parser.add_argument('--prediction_file', default=None, help='path to the prediction file')
	parser.add_argument('--target_file', default=None, help='path to the target file')
	# evaluate files in one directory
	parser.add_argument('--prediction_dir', default=None, help='directory of prediction files')
	parser.add_argument('--target_dir', default=None, help='directory of target files')
	# set stats_file for saving results when evaluating files in one directory
	parser.add_argument('--stats_file', default=None, help='path to the file saving evaluation results')
	parser.add_argument('--overwrite', action="store_true", help='overwrite previous stats_file')

	opts = parser.parse_args()

	evaluate_function(**vars(opts))

if __name__ == '__main__':
	main()