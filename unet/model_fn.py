import tensorflow as tf
import os
import tifffile
import sys
import numpy as np

from unet.network import UNet, _3d_to_2d, get_loss
from unet.data import input_function, input_function_npz
from unet.data import prepare_test



"""This script trains or evaluates the model.
"""


class Model(object):

	def __init__(self, opts, conf_unet):
		self.opts = opts
		self.conf_unet = conf_unet

	def _model_fn(self, features, labels, mode):
		"""Initializes the Model representing the model layers
		and uses that model to build the necessary EstimatorSpecs for
		the `mode` in question. For training, this means building losses,
		the optimizer, and the train op that get passed into the EstimatorSpec.
		For evaluation and prediction, the EstimatorSpec is returned without
		a train op, but with the necessary parameters for the given mode.

		Args:
			features: tensor representing input images
			labels: tensor representing class labels for all input images
			mode: current estimator mode; should be one of
				`tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`

		Returns:
			tf.estimator.EstimatorSpec
		"""
		if self.opts.proj_model:
			projection = _3d_to_2d(self.conf_unet)
			features = projection(features, mode == tf.estimator.ModeKeys.TRAIN)
			self.conf_unet['dimension'] = '2D'
		network = UNet(self.conf_unet)
		outputs = network(features, mode == tf.estimator.ModeKeys.TRAIN)

		predictions = {'pixel_values': tf.add(features, outputs) if self.opts.offset else outputs}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate the MSE loss.
		loss = get_loss(labels, features, outputs, self.opts, self.conf_unet)

		# Create a tensor named MSE for logging purposes.
		tf.identity(loss, name=self.opts.loss_type)
		tf.summary.scalar(self.opts.loss_type, loss)

		if mode == tf.estimator.ModeKeys.TRAIN:
			global_step = tf.train.get_or_create_global_step()

			optimizer = tf.train.AdamOptimizer(
							learning_rate=self.opts.learning_rate,
							beta1=0.5,
							beta2=0.999)

			# Batch norm requires update ops to be added as a dependency to train_op
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer.minimize(loss, global_step)
		else:
			train_op = None

		return tf.estimator.EstimatorSpec(
					mode=mode,
					predictions=predictions,
					loss=loss,
					train_op=train_op)

	def train(self):
		# Using the Winograd non-fused algorithms provides a small performance boost.
		os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

		session_config = tf.ConfigProto()
		session_config.gpu_options.allow_growth = True

		run_config = tf.estimator.RunConfig().replace(
						save_checkpoints_steps=self.opts.save_checkpoints_iter,
						keep_checkpoint_max=0,
						session_config=session_config)

		classifier = tf.estimator.Estimator(
						model_fn=self._model_fn,
						model_dir=self.opts.model_dir,
						config=run_config)

		tensors_to_log = {self.opts.loss_type: self.opts.loss_type}
		logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

		input_function = input_function if not self.opts.npz else input_function_npz
		def input_fn_train():
			return input_function(opts=self.opts, mode='train')

		print('Start training...')
		classifier.train(input_fn=input_fn_train, hooks=[logging_hook])

	def predict(self):
		# Using the Winograd non-fused algorithms provides a small performance boost.
		os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

		classifier = tf.estimator.Estimator(
						model_fn=self._model_fn,
						model_dir=self.opts.model_dir)

		def input_fn_predict():
			return input_function(opts=self.opts, mode='pred')

		checkpoint_file = os.path.join(self.opts.model_dir, 'model.ckpt-'+str(self.opts.checkpoint_num))
		preds = classifier.predict(input_fn=input_fn_predict, checkpoint_path=checkpoint_file)

		model_name = self.opts.model_dir.split('/')[-1]
		pred_result_dir = os.path.join(self.opts.result_dir, model_name)
		if not os.path.exists(pred_result_dir):
			os.makedirs(pred_result_dir)
		pred_result_dir = os.path.join(pred_result_dir, 'checkpoint_%s' % str(self.opts.checkpoint_num))
		if not os.path.exists(pred_result_dir):
			os.makedirs(pred_result_dir)
		pred_result_dir = os.path.join(pred_result_dir, 'patch%s_step%s' \
				% (str(self.opts.test_patch_size), str(self.opts.test_step_size)))
		if not os.path.exists(pred_result_dir):
			os.makedirs(pred_result_dir)

		print('Start predicting...')
		if sum(self.opts.test_patch_size) == 0 and sum(self.opts.test_step_size) == 0:
			for idx, pred in enumerate(preds):
				print('Process testing sample %d' % idx)
				pred_result = pred['pixel_values'][:, :, :, 0]
				path_tiff = os.path.join(pred_result_dir, 'prediction_{:02d}.tiff'.format(idx))
				tifffile.imsave(path_tiff, pred_result)
				print('saved:', path_tiff)
		else:
			for idx in range(0, self.opts.num_test_pairs):
				print('Process testing sample %d' % idx)
				path_tiff = os.path.join(self.opts.result_dir, 'signal_{:02d}.tiff'.format(idx))
				signal = tifffile.imread(path_tiff)
				patch_ids, real_patch_size = prepare_test(signal, self.opts.test_patch_size, self.opts.test_step_size)
				predictions = {}
				for i, pred in enumerate(preds):
					patch = patch_ids[i]
					print('Step {:d}/{:d} processing results for ({:d},{:d},{:d})'.format(
								i+1, len(patch_ids), patch[0], patch[1], patch[2]),
								end='\r',
								flush=True)
					pixel_values = pred['pixel_values']

					for j in range(real_patch_size[0]):
						for k in range(real_patch_size[1]):
							for l in range(real_patch_size[2]):
								key = (patch[0]+j, patch[1]+k, patch[2]+l)
								if key not in predictions.keys():
									predictions[key] = []
								predictions[key].append(pixel_values[j, k, l, 0])

					if i == len(patch_ids) - 1:
						pred_result = np.zeros(signal.shape, dtype=np.float32)
						for key in predictions.keys():
							pred_result[key[0],	key[1], key[2]] = np.mean(predictions[key])

						path_tiff = os.path.join(pred_result_dir, 'prediction_{:02d}.tiff'.format(idx))
						tifffile.imsave(path_tiff, pred_result)
						print('saved:', path_tiff)
						break

		print('Done.')
		sys.exit(0)
