import os
import argparse
import tensorflow as tf

from unet import Model
from network_configure import conf_unet

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_id', type=int, help='which gpu to use')
	# testing data
	parser.add_argument('--tiff_dataset_dir', type=str, help='directory of tiff files corresponding to testing data')
	parser.add_argument('--num_test_pairs', type=int, default=20, help='number of pairs for testing')
	# testing settings
	parser.add_argument('--result_dir', type=str, help='directory of resulted tiff files')
	parser.add_argument('--model_dir', default='saved_models', help='directory to save settings and checkpoints \
						of model during training')
	parser.add_argument('--checkpoint_num', type=str, default='75000', help='which checkpoint is used for validation/prediction')
	# options for cropped prediction
	parser.add_argument('--cropped_prediction', action="store_true", help='whether to crop during prediction')
	parser.add_argument('--predict_patch_size', nargs='+', type=int, default=[128, 256, 256], help='patch size \
						for cropped prediction, only used when cropped_prediction is True')
	parser.add_argument('--overlap', type=int, default=(64, 64, 64), help='overlap size between two consecutive \
						patches, only used when cropped_prediction is True')
	parser.add_argument('--test_batch_size', type=int, default=1, help='size of each batch, only used when \
						cropped_prediction is True')
	# extra options
	parser.add_argument('--CARE_normalize', action="store_true", help='whether to apply percentile normalization \
						to the input images')
	parser.add_argument('--proj_model', action="store_true", help='whether to use ProjectionNet to project \
						3D images to 2D, used in 3D-to-2D transform task, e.g. Flywings projection')
	parser.add_argument('--offset', action="store_true", help='whether to add inputs to the outputs')
	
	opts = parser.parse_args()
        
	os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
	model = Model(opts, conf_unet)
	model.predict()

if __name__ == '__main__':
	# Choose which gpu or cpu to use
	tf.logging.set_verbosity(tf.logging.ERROR)
	tf.app.run()