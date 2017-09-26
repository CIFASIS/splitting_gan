import tflib as lib

import numpy as np
import tensorflow as tf

def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None, n_labels=None, extensible=False):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if axes != [0,2,3]:
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = mean.get_shape().as_list() # shape is [1,n,1,1]
    if extensible:
        offset_m = lib.extensible_param(name + '.offset', np.zeros([n_labels, shape[1]], dtype='float32'))
        scale_m = lib.extensible_param(name + '.scale', np.ones([n_labels, shape[1]], dtype='float32'))
    else:
        offset_m = lib.param(name+'.offset', np.zeros([n_labels,shape[1]], dtype='float32'))
        scale_m = lib.param(name+'.scale', np.ones([n_labels,shape[1]], dtype='float32'))
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)
    return result, offset_m, scale_m

def BatchnormB(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None, n_labels=None, n_start_labels=None):
    """ based on conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if axes != [0,2,3]:
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = mean.get_shape().as_list() # shape is [1,n,1,1]
    init_scale = np.zeros([n_labels, shape[1]], dtype='float32')
    init_scale[:n_start_labels] = 1.0
    offset_m = lib.param(name + '.offset', np.zeros([n_labels, shape[1]], dtype='float32'))
    scale_m = lib.param(name + '.scale', init_scale)
    offset = tf.matmul(labels, offset_m)
    scale = tf.matmul(labels, scale_m)
    result = tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)
    return result, offset_m, scale_m
