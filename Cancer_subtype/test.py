from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import sys
import os
from Run_StainSep import run_stainsep
from Run_ColorNorm import run_colornorm, run_batch_colornorm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
listr = tf.config.experimental.list_physical_devices(device_type=None)
print(listr)
