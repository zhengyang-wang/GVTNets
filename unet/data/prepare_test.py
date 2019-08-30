import numpy as np

def prepare_test(data, patch_size, step_size):
	# Determine patches for testing.
	patch_ids = []

	D, H, W = data.shape

	assert 0 <= patch_size[0] <= D, 'The patch size should be smaller than or equal to the data size.'
	assert 0 <= patch_size[1] <= H, 'The patch size should be smaller than or equal to the data size.'
	assert 0 <= patch_size[2] <= W, 'The patch size should be smaller than or equal to the data size.'
	assert 0 <= step_size[0] <= patch_size[0], 'The step size should be smaller than or equal to the patch size.'
	assert 0 <= step_size[1] <= patch_size[1], 'The step size should be smaller than or equal to the patch size.'
	assert 0 <= step_size[2] <= patch_size[2], 'The step size should be smaller than or equal to the patch size.'
	
	# zero means the patch size is set to the data size
	patch_size_D = D if patch_size[0] == 0 else patch_size[0]
	patch_size_H = H if patch_size[1] == 0 else patch_size[1]
	patch_size_W = W if patch_size[2] == 0 else patch_size[2]
	# zero means the step size is set to the patch size
	step_size_D = patch_size_D if step_size[0] == 0 else step_size[0]
	step_size_H = patch_size_H if step_size[1] == 0 else step_size[1]
	step_size_W = patch_size_W if step_size[2] == 0 else step_size[2]

	drange = list(range(0, D-patch_size_D+1, step_size_D))
	hrange = list(range(0, H-patch_size_H+1, step_size_H))
	wrange = list(range(0, W-patch_size_W+1, step_size_W))

	if (D-patch_size_D) % step_size_D != 0:
		drange.append(D-patch_size_D)
	if (H-patch_size_H) % step_size_H != 0:
		hrange.append(H-patch_size_H)
	if (W-patch_size_W) % step_size_W != 0:
		wrange.append(W-patch_size_W)

	for d in drange:
		for h in hrange:
			for w in wrange:
				patch_ids.append((d, h, w))

	print ('Data shape:', data.shape, 'Number of testing patches:', len(patch_ids))
	return patch_ids, [patch_size_D, patch_size_H, patch_size_W]