import os
import argparse
import tensorflow as tf
import json

from unet import Model
from network_configure import conf_basic_ops, conf_attn_same, conf_attn_up, conf_attn_down, conf_unet

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--npz_dataset_dir', type=str, help='directory of npz files, only used with --npz')
	parser.add_argument('--cropped', action="store_true", help='whether input dataset is already cropped')
	parser.add_argument('--save_tfrecords', action="store_true", help='whether save and use tfrecord files')
	parser.add_argument('--tf_dataset_dir', type=str, help='directory of tfrecord files')
	parser.add_argument('--num_train_pairs', type=int, default=30, help='number of pairs for training')
	parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
	parser.add_argument('--buffer_size', type=int, default=30, help='number of images to cache in memory')
	parser.add_argument('--loss_type', type=str, default='MSE', help='meam squared error or mean absolute error loss')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--num_iters', type=int, default=500, help='number of training iterations')
	parser.add_argument('--save_checkpoints_iter', type=int, default=500, help='iterations at which to save checkpoints of model')
	parser.add_argument('--model_dir', default='saved_models', help='base directory for saved models')
	parser.add_argument('--patch_size', nargs='+', type=int, default=[32, 64, 64], help='size of patches to sample from Dataset elements')
	parser.add_argument('--gpu_id', type=int, help='which gpu to use')
	parser.add_argument('--num_parallel_calls', type=int, default=5, help='number of records that are processed in parallel')
	parser.add_argument('--proj_model', action="store_true", help='whether model project 3D images to 2D, e.g. Flywings projection')
	parser.add_argument('--offset', action="store_true", help='whether add inputs to the network output, used in CARE')
	parser.add_argument('--probalistic', action="store_true", help='train with probalistic loss')
	opts = parser.parse_args()

	if not os.path.exists(opts.model_dir):
		os.makedirs(opts.model_dir)
	else:
		print("The model directory already exists!")
		return 0

	with open(os.path.join(opts.model_dir, 'train_settings.json'), 'w') as f:
		json.dump(vars(opts), f, indent=4)
		json.dump(conf_basic_ops, f, indent=4)
		json.dump(conf_attn_same, f, indent=4)
		json.dump(conf_attn_up, f, indent=4)
		json.dump(conf_attn_down, f, indent=4)
		json.dump(conf_unet, f, indent=4)

	os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
	model = Model(opts, conf_unet)
	model.train()

if __name__ == '__main__':
	# Choose which gpu or cpu to use
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
