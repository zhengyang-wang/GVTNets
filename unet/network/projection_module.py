import tensorflow as tf
from tensorflow.layers import max_pooling3d
from basic_ops import *
from attention_module import *

"""This script defines projection module only used for the 3D to 2D projection tasks.
By default, the bottom block and the upsample blocks are attention-based.
"""

class _3d_to_2d(object):
    def __init__(self, conf_unet):

        self.depth = conf_unet['depth']
        self.bottom_block = conf_unet['bottom_block']
        self.upsampling = conf_unet['upsampling']


    def __call__(self, inputs, training):
        """3D to 2D projection module.
        Args:
            inputs: a tensor. The input 3D images of shape [batch, d, h, w, channels (1)]
            training: a boolean for batch normalization and dropout.

        Retuen:
            The tensor of 2D projected intermediate image of shape [batch, h, w, channels (1)]
        """
        return self._build_network(inputs, training)

    
    def _get_bottom_function(self, name):
        if name == 'same_gto':
            return same_gto
        elif name == 'res_block':
            return res_block
        else:
            raise ValueError("Unsupported function: %s." % (name))
            

    def _get_upsampling_function(self, name):
        if name == 'up_gto_v1':
            return up_gto_v1
        elif name == 'up_gto_v2':
            return up4_gto_v2
        elif name == 'transposed_convolution':
            return up_transposed_convolution
        else:
            raise ValueError("Unsupported function: %s." % (name))
    
    
    def _build_network(self, inputs, training):
        with tf.variable_scope('projection'):
            outputs = inputs
            # encoding_blocks
            for n in range(self.depth-1):
                if n > 0:
                    outputs = batch_norm(outputs, training, 'BN_down_%d' % n)
                    outputs = relu(outputs, 'Relu_down_%d' % n)
                outputs = convolution_3D(outputs, 8, (3,5,5), 1, True, 'Conv_down_%d' % n)
                outputs = max_pooling3d(outputs, (1,4,4), (1,4,4), name='Downsample_%d' % n)
            
            # bottom_blocks
            for n, bottom in enumerate(conf_unet['bottom_block']):
                bottom_block = _get_bottom_function(bottom)
                conv_path = batch_norm(outputs, training, 'BN_bottom_%d' % n)
                conv_path = relu(conv_path, 'Relu_bottom_%d' % n)
                conv_path = convolution_3D(conv_path, 8, (3,5,5), 1, True, 'Conv_bottom_%d' % n)
                outputs = bottom_block(outputs, 8, training, '3D', 'Bottom_%d' % n)
                outputs = outputs + conv_path
            
            # decoding_blocks
            for n, upsample in reversed(list(enumerate(conf_unet['upsampling']))):
                upsample_block = _get_upsampling_function(upsample)
                outputs = upsample_block(outputs, 8, training, '3D', 'Upsample_%d' % n)
                outputs = batch_norm(outputs, training, 'BN_up_%d' % n)
                outputs = relu(outputs, 'Relu_up_%d' % n)
                outputs = convolution_3D(outputs, 8, (3,5,5), 1, True, 'Conv_up_%d' % n)
                
            # projection
            outputs = batch_norm(outputs, training, 'BN_up_%d' % n)
            outputs = relu(outputs, 'Relu_up_%d' % n)
            outputs = convolution_3D(outputs, 1, (5,5,5), 1, True, 'Conv_final')
            outputs = tf.multiply(outputs, inputs)
            outputs = tf.squeeze(outputs, -1)
            outputs = tf.transpose(outputs, [0,2,3,1])
            outputs = tf.reduce_mean(outputs, -1, keepdims=True)
                    
        return outputs

