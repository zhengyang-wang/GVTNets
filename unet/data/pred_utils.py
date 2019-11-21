from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import warnings
import numpy as np


def _raise(e):
    raise e

# Inheritted from CARE
class PercentileNormalizer(object):

    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=np.float32, **kwargs):

        (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100) or _raise(ValueError())
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs

    def before(self, img, axes):

        len(axes) == img.ndim or _raise(ValueError())
        channel = None if axes.find('C')==-1 else axes.find('C')
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img,self.pmin,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        self.ma = np.percentile(img,self.pmax,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        return (img-self.mi)/(self.ma-self.mi+1e-20)

    def after(self, img):

        self.do_after or _raise(ValueError())
        alpha = self.ma - self.mi
        beta  = self.mi
        return ( alpha*img+beta ).astype(self.dtype,copy=False)

    def do_after(self):
        
        return self._do_after
    
    
# Inheritted from CARE
class PadAndCropResizer(object):

    def __init__(self, mode='reflect', **kwargs):

        self.mode = mode
        self.kwargs = kwargs
        
    def _normalize_exclude(self, exclude, n_dim):
        """Return normalized list of excluded axes."""
        if exclude is None:
            return []
        exclude_list = [exclude] if np.isscalar(exclude) else list(exclude)
        exclude_list = [d%n_dim for d in exclude_list]
        len(exclude_list) == len(np.unique(exclude_list)) or _raise(ValueError())
        all(( isinstance(d,int) and 0<=d<n_dim for d in exclude_list )) or _raise(ValueError())
        return exclude_list

    def before(self, x, div_n, exclude):

        def _split(v):
            a = v // 2
            return a, v-a
        exclude = self._normalize_exclude(exclude, x.ndim)
        self.pad = [_split((div_n-s%div_n)%div_n) if (i not in exclude) else (0,0) for i,s in enumerate(x.shape)]
        x_pad = np.pad(x, self.pad, mode=self.mode, **self.kwargs)
        for i in exclude:
            del self.pad[i]
        return x_pad

    def after(self, x, exclude):

        pads = self.pad[:len(x.shape)]
        crop = [slice(p[0], -p[1] if p[1]>0 else None) for p in self.pad]
        for i in self._normalize_exclude(exclude, x.ndim):
            crop.insert(i,slice(None))
        len(crop) == x.ndim or _raise(ValueError())
        return x[tuple(crop)]
    

class PatchPredictor(object):
    
    def __init__(self, patch_size, overlap, proj=False):
        self.patch_size = np.array(patch_size)
        self.overlap = np.array(overlap)
        self.proj = proj
        
    def before(self, img, div_n):
        self.shape = img.shape[:-1]
        self.patch_size = (np.ceil(np.array(self.patch_size)/div_n)*div_n).astype('int')
        if self.proj:
            assert self.patch_size[0]>=self.shape[0]
        patch_list = [img[src_s] for src_s in self._get_crop_coord()]
        self.npatches = len(patch_list)
        return np.stack(patch_list)
    
    def after(self, patches):
        assert len(patches) == self.npatches
        padded_patches = np.zeros((self.npatches,)+(self.shape[1:] if self.proj else self.shape)+(1,))
        mask = np.zeros((self.npatches,)+(self.shape[1:] if self.proj else self.shape)+(1,))
        for idx, src_s in enumerate(self._get_crop_coord()):
            src_s = src_s[1:] if self.proj else src_s
            padded_patches[idx][src_s] = patches[idx]
            mask[idx][src_s] = 1
        return np.average(padded_patches, axis=0, weights=mask)
    
    def _get_crop_coord(self):
        shape = self.shape
        size = np.where(shape<self.patch_size, shape, self.patch_size)
        assert len(shape)==len(size)
        margin = np.where(shape>=size, self.overlap, 0)

        n_tiles = (shape-size-1)//(size-margin)+1
        if len(n_tiles)==3:
            n_tiles_z, n_tiles_y, n_tiles_x = n_tiles
        elif len(n_tiles)==2:
            n_tiles_z, (n_tiles_y, n_tiles_x) = 0, n_tiles

        for i in range(n_tiles_z+1):
            if len(n_tiles)==3:
                src_start_z = i*(size-margin)[0] if i<n_tiles_z else (shape-size)[0]
                src_end_z = src_start_z+size[0]

            for j in range(n_tiles_y+1):
                src_start_y = j*(size-margin)[-2] if j<n_tiles_y else (shape-size)[-2]
                src_end_y = src_start_y+size[1]

                for k in range(n_tiles_x+1):
                    src_start_x = k*(size-margin)[-1] if k<n_tiles_x else (shape-size)[-1]
                    src_end_x = src_start_x+size[-1]

                    if len(n_tiles)==3:
                        src_s = (slice(src_start_z, src_end_z), 
                                 slice(src_start_y, src_end_y), 
                                 slice(src_start_x, src_end_x))

                    elif len(n_tiles)==2:
                        src_s = (slice(src_start_y, src_end_y), 
                                 slice(src_start_x, src_end_x))

                    yield src_s