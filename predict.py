import os
import argparse
import tensorflow as tf

from unet import Model
from network_configure import conf_unet

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--tf_dataset_dir', type=str, help='directory of tfrecord files')
	parser.add_argument('--num_test_pairs', type=int, default=20, help='number of pairs for testing')
	parser.add_argument('--test_patch_size', nargs='+', type=int, default=[0, 0, 0], help='size of patches to cut Dataset elements during testing.\
							zero means the patch size is set to the data size')
	parser.add_argument('--test_step_size', nargs='+', type=int, default=[0, 0, 0], help='size of step between patches during testing.\
							zero means the step size is set to the patch size')
	parser.add_argument('--result_dir', type=str, help='directory of resulted tiff files')
	parser.add_argument('--batch_size', type=int, default=1, help='size of each batch')
	parser.add_argument('--model_dir', default='saved_models', help='base directory for saved models')
	parser.add_argument('--checkpoint_num', type=int, default=49999, help='which checkpoint is used for validation/prediction')
	parser.add_argument('--gpu_id', type=int, help='which gpu to use')
	parser.add_argument('--num_parallel_calls', type=int, default=5, help='number of records that are processed in parallel')
	opts = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
	model = Model(opts, conf_unet)
	model.predict()

if __name__ == '__main__':
	# Choose which gpu or cpu to use
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()