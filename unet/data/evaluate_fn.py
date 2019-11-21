import os, pickle, glob, fnmatch
import pandas as pd
import numpy as np
import scipy.stats as stats

from tqdm import tqdm
from tifffile import imread
from sklearn.metrics import r2_score
from scipy.misc import ascent
from skimage.measure import compare_psnr, compare_mse, compare_ssim

def evaluate_function(prediction_file=None, target_file=None, prediction_dir=None, target_dir=None, stats_file=None, overwrite=True):
	if prediction_file != None and target_file != None:
		print('Evaluate the prediction: %s' % prediction_file)
		predictions = [prediction_file]
		targets = [target_file]
		save_stats = False
	elif prediction_dir != None and target_dir != None:
		print('Evaluate the predictions in %s' % prediction_dir)
		predictions = sorted([os.path.join(prediction_dir, f) for f in os.listdir(prediction_dir) if 
				fnmatch.fnmatch(f, '*.tif*')])
		targets = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if 
				fnmatch.fnmatch(f, '*.tif*')])
		save_stats = True
	else:
		print('You need to specify either prediction_file + target_file or prediction_dir + target_dir + stats_file!')
		return -1

	assert len(predictions) == len(targets), ('Length of predictions %d does not match length of targets %d'
                                              %(len(predictions),len(targets)))
	for i in range(len(predictions)):
		assert os.path.basename(predictions[i]).strip('pred_') == os.path.basename(targets[i]), \
				'Prediction-Target pairs do not match!'

	if save_stats and not overwrite and os.path.exists(stats_file):
		stats_all, stats_per_im = pickle.load(open(stats_file, "rb"))
		print(stats_all)
	else:
		stats_per_im, stats_all = eval_images(targets, predictions)
		if save_stats:
			pickle.dump([stats_all, stats_per_im], open(stats_file, "wb"))
			print(stats_all[['r','ssim','nrmse','psnr']])
		else:
			print(stats_all['r','ssim','nrmse','psnr'])

def eval_images(path_targets, path_preds):
	log_per_im = list()
	im_preds = list()
	im_targets = list()
	
	pbar = tqdm(zip(range(len(path_preds)), path_preds, path_targets))
	for i, path_pred, path_target in pbar:
		im_pred = imread(path_pred)
		im_target = imread(path_target)

		im_preds.append(im_pred)
		im_targets.append(im_target)

		err_map, n_pixels, stats = get_stats(im_pred, im_target)
		stats['img'] = i
		df_per_im = pd.DataFrame.from_dict([stats])
		log_per_im.append(df_per_im)
	log_per_im = pd.concat(log_per_im)
		
# 	im_pred_all_flat = np.hstack([im.flatten() for im in im_preds])
# 	im_target_all_flat = np.hstack([im.flatten() for im in im_targets])
# 	err_map, n_pixels, stats = get_stats(im_pred_all_flat, im_target_all_flat)
# 	log_all = pd.DataFrame.from_dict([stats])
	log_all = log_per_im.mean(axis=0)

	return log_per_im, log_all

def get_stats(pred, target):
	nrmse, psnr, ssim = get_scores(target, pred)
    
	delta = pred - target
	err_map = (delta)**2
	se = np.sum(err_map)
	n_pixels = np.prod(target.shape)
	mse = np.mean(err_map)

	target_flat = target.flatten()
	pred_flat = pred.flatten()
	
	R2 = r2_score(target_flat, pred_flat)
	
	y_bar = np.mean(target_flat)
	
	denom = np.sum((target_flat - y_bar)**2)
	nom = np.sum((pred_flat - target_flat)**2)
	
	exp_var = 1 - (nom / denom)
	
	r = stats.pearsonr(target_flat, pred_flat)[0]
	
	var_pred = np.var(pred_flat)
	var_target = np.var(target_flat)
	
	delta_min = np.min(delta) 
	delta_max = np.max(delta)
    
	all_stats = {'n_pixels':  n_pixels, 'mse': mse, 'R2': R2, \
			 'nrmse': nrmse, 'psnr': psnr, 'ssim': ssim, \
			 'r': r, 'exp_var': exp_var, \
			 'var_pred': var_pred, 'var_target': var_target, \
			 'delta_min': delta_min, 'delta_max': delta_max}
	
	return err_map, se, all_stats


def get_scores(gt, x, multichan=False):
    
	gt_, x_ = norm_minmse(gt, x)
    
	mse = compare_mse(gt_, x_)
	psnr = compare_psnr(gt_, x_, data_range = 1.)
	ssim = compare_ssim(gt_, x_, data_range = 1., multichannel=multichan)
    
	return np.sqrt(mse), psnr, ssim


def norm_minmse(gt, x, normalize_gt=True):
	if normalize_gt:
		gt = normalize(gt, 0.1, 99.9, clip=False).astype(np.float32, copy=False)
	x = x.astype(np.float32, copy=False) - np.mean(x)
	gt = gt.astype(np.float32, copy=False) - np.mean(gt)
	scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
	return gt, scale * x


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
	mi = np.percentile(x, pmin, axis=axis, keepdims=True)
	ma = np.percentile(x, pmax, axis=axis, keepdims=True)
	return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
	if dtype is not None:
		x = x.astype(dtype,copy=False)
		mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
		ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
		eps = dtype(eps)

	try:
		import numexpr
		x = numexpr.evaluate("(x - mi) / (ma - mi + eps)")
	except ImportError:
		x = (x - mi) / (ma - mi + eps)

	if clip:
		x = np.clip(x, 0, 1)

	return x