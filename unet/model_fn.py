import tensorflow as tf
import os
import tifffile
import sys
import numpy as np

from unet.network import UNet, ProjectionNet, get_loss
from unet.data import input_function


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
			projection = ProjectionNet(self.conf_unet)
			features = projection(features, mode == tf.estimator.ModeKeys.TRAIN)
			assert self.conf_unet['dimension'] == '2D'
		network = UNet(self.conf_unet)
		outputs = network(features, mode == tf.estimator.ModeKeys.TRAIN)

		# If set opts.offset true, outputs will be considered as an offset to inputs
		predictions = {'pixel_values': tf.add(features, outputs) if self.opts.offset else outputs}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate the loss.
		loss = get_loss(labels, features, outputs, self.opts.loss_type,
					self.opts.probalistic, self.opts.offset, self.conf_unet['dimension'])

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

		transformer = tf.estimator.Estimator(
						model_fn=self._model_fn,
						model_dir=self.opts.model_dir,
						config=run_config)

		tensors_to_log = {self.opts.loss_type: self.opts.loss_type}
		logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

		train_input_fn = input_function(opts=self.opts, mode='train')

		print('Start training...')
		transformer.train(input_fn=train_input_fn, hooks=[logging_hook])

	def predict(self):
		pred_result_dir = os.path.join(self.opts.result_dir, 'checkpoint_%s' % str(self.opts.checkpoint_num))
		if not os.path.exists(pred_result_dir):
			os.makedirs(pred_result_dir)
		else:
			print('The result dir for checkpoint_num %d already exist.' % self.opts.checkpoint_num)
			return 0

		# Using the Winograd non-fused algorithms provides a small performance boost.
		os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

		transformer = tf.estimator.Estimator(
						model_fn=self._model_fn,
						model_dir=self.opts.model_dir)

		predict_input_fns = input_function(opts=self.opts, mode='predict')

		checkpoint_file = os.path.join(self.opts.model_dir, 'model.ckpt-'+str(self.opts.checkpoint_num))

		for idx, predict_input_fn in enumerate(predict_input_fns):
			print('Predicting testing sample %d ...' % idx)
			prediction = transformer.predict(input_fn=predict_input_fn, checkpoint_path=checkpoint_file)

			if self.opts.cropped_prediction:
				pass

			else:
				# Take the entire image as the input and make predictions.
				# If the image is very large, set --gpu_id to -1 to use cpu mode.
				prediction = list(prediction)
				prediction = prediction[0]['pixel_values'][:, :, :, 0]
				path_tiff = os.path.join(pred_result_dir, 'prediction_{:02d}.tiff'.format(idx))
				tifffile.imsave(path_tiff, prediction)
				print('saved:', path_tiff)

		print('Done.')
		sys.exit(0)
