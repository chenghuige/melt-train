import tensorflow as tf

_auc_module = tf.load_op_library('auc.so')
auc = _auc_module.auc
