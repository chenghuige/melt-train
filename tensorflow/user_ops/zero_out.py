import tensorflow as tf

_zero_out_module = tf.load_op_library('zero_out_op_kernel_1.so')
zero_out = _zero_out_module.zero_out



 
