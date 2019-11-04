import tensorflow as tf
import os
import tifffile
import sys
import numpy as np

from unet.network import UNet, ProjectionNet, get_loss
from unet.data import train_input_function, pred_input_function, load_testing_tiff
from unet.data import PercentileNormalizer, PadAndCropResizer, PatchPredictor

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
		features = tf.cast(features, tf.float32)
		if self.opts.proj_model:
			projection = ProjectionNet(self.conf_unet)
			features = projection(features, mode == tf.estimator.ModeKeys.TRAIN)
			self.conf_unet['dimension'] = '2D'
		network = UNet(self.conf_unet)
		outputs, penult = network(features, mode == tf.estimator.ModeKeys.TRAIN)
		outputs = tf.add(features, outputs) if self.opts.offset else outputs

		# If set opts.offset true, outputs will be considered as an offset to inputs
		predictions = {'pixel_values': outputs}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate the loss.
		loss = get_loss(labels, outputs, penult, self.opts.loss_type,
					self.opts.probalistic, self.opts.offset, self.conf_unet['dimension'])

		# Create a tensor named MSE/MAE for logging purposes.
		if self.opts.probalistic:
			# Exclude uncertainty terms in logged MSE/MAE
			true_loss = get_loss(labels, outputs, penult, self.opts.loss_type, False,
				self.opts.offset, self.conf_unet['dimension'])
			tf.identity(true_loss, name=self.opts.loss_type)
			tf.summary.scalar(self.opts.loss_type, true_loss)
		else:
			tf.identity(loss, name=self.opts.loss_type)
			tf.summary.scalar(self.opts.loss_type, loss)

		if mode == tf.estimator.ModeKeys.TRAIN:
			global_step = tf.train.get_or_create_global_step()
			learning_rate = tf.train.exponential_decay(self.opts.learning_rate, global_step, 
				self.opts.lr_decay_steps, self.opts.lr_decay_rate, True)
			optimizer = tf.train.AdamOptimizer(
							learning_rate=learning_rate)

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

		print('Start training...')
		transformer.train(input_fn=train_input_function(opts=self.opts), hooks=[logging_hook])

	def predict(self):
		sources, fnames = load_testing_tiff(self.opts.tiff_dataset_dir, self.opts.num_test_pairs)
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

		checkpoint_file = os.path.join(self.opts.model_dir, 'model.ckpt-'+str(self.opts.checkpoint_num))

		resizer = PadAndCropResizer()
		cropper = (PatchPredictor(self.opts.patch_size, self.opts.overlap, self.opts.proj_model) if 
			self.opts.cropped_prediction else None)
		normalizer = PercentileNormalizer() if self.opts.CARE_normalize else None
		div_n = (4 if self.opts.proj_model else 2)**(self.conf_unet['depth']-1)
		excludes = ([3,0], 2) if self.opts.proj_model else (3,3)

		for idx, source in enumerate(sources):
			print('Predicting testing sample %d, shape %s ...' % (idx, str(source.shape)))
			source = normalizer.before(source, 'ZYXC') if self.opts.CARE_normalize else source
			source = resizer.before(source, div_n=div_n, exclude=excludes[0])

			if self.opts.cropped_prediction:
				patches = cropper.before(source, div_n)
				prediction = transformer.predict(
					input_fn=pred_input_function(self.opts, patches), checkpoint_path=checkpoint_file)
				prediction = np.stack([pred['pixel_values'] for pred in prediction])
				prediction = cropper.after(prediction)
			else:
				# Take the entire image as the input and make predictions.
				# If the image is very large, set --gpu_id to -1 to use cpu mode.
				prediction = transformer.predict(
					input_fn=pred_input_function(self.opts, source[None]), checkpoint_path=checkpoint_file)
				prediction = list(prediction)[0]['pixel_values']

			prediction = resizer.after(prediction, exclude=excludes[1])
			prediction = (normalizer.after(prediction) if
				self.opts.normalize and normalizer.do_after() else prediction)
			prediction = prediction[0] if self.opts.proj_model else prediction
			path_tiff = os.path.join(pred_result_dir, 'pred_'+fnames[idx])
			tifffile.imsave(path_tiff, prediction[..., 0])
			print('saved:', path_tiff)

		print('Done.')
		sys.exit(0)

        