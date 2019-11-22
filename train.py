import os
import argparse
import tensorflow as tf
import json

from unet import Model
from network_configure import conf_basic_ops, conf_attn_same, conf_attn_up, conf_attn_down, conf_unet

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_id', type=int, help='which gpu to use')
	# training data
	parser.add_argument('--npz_dataset_dir', type=str, help='directory of npz files corresponding to training data')
	parser.add_argument('--already_cropped', action="store_true", help='whether training data are already cropped')
	parser.add_argument('--train_patch_size', nargs='+', type=int, help='size of training patches after cropping: [(D,) H, W]')
	parser.add_argument('--num_train_pairs', type=int, default=30, help='number of pairs for training. \
																		Only used when --opts.already_cropped is NOT set.')
	parser.add_argument('--save_tfrecords', action="store_true", help='whether to save and use tfrecord files for faster I/O during training')
	parser.add_argument('--tf_dataset_dir', type=str, help='directory of tfrecord files when --save_tfrecords')
	# training settings
	parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
	parser.add_argument('--loss_type', type=str, default='MSE', help='meam squared error (MSE) or mean absolute error (MAE) loss')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='base learning rate')
	parser.add_argument('--lr_decay_rate', type=float, default=1.0, help='learning rate decay rate. Set to 1.0 to disable learning rate decay.')
	parser.add_argument('--lr_decay_steps', type=int, default=10000, help='number of training iterations per learning rate decay')
	parser.add_argument('--num_iters', type=int, default=500, help='number of training iterations (each iteration processes one batch)')
	# training checkpoints
	parser.add_argument('--save_checkpoints_iter', type=int, default=500, help='iterations at which to save training checkpoints of model')
	parser.add_argument('--model_dir', default='saved_models', help='directory to save settings and checkpoints of model during training')
	# extra options
	parser.add_argument('--proj_model', action="store_true", help='whether to use ProjectionNet to project 3D images to 2D, \
																	used in 3D-to-2D transform task, e.g. Flywings projection')
	parser.add_argument('--offset', action="store_true", help='whether to add inputs to the outputs')
	parser.add_argument('--probalistic', action="store_true", help='whether to train with probalistic loss')
	
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
