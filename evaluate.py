import os
import argparse

from unet.data import evaluate_function

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--prediction_file', default=None, help='path to the prediction file')
	parser.add_argument('--target_file', default=None, help='path to the target file')
	parser.add_argument('--prediction_dir', default=None, help='directory of prediction files')
	parser.add_argument('--target_dir', default=None, help='directory of target files')
	parser.add_argument('--overwrite', type=str2bool, default=True, help='overwrite previous results')
	opts = parser.parse_args()

	evaluate_function(**vars(opts))

if __name__ == '__main__':
	main()