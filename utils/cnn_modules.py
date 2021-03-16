# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers.merge import concatenate
# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out, act):
	"""
	layer_in: input data shape
    f1: # of filters in 1x1 conv
	f2_in, f2_out: # of filters in the consecutive 1x1 conv and 3x3 conv layers
	f3_in, f3_out: # of filters in the consecutive 1x1 conv and 5x5 conv layers
	f4_out: # of filters in the conv layer after MaxPool layer

	Example:
    inputs = tf.keras.Input(shape=(256, 256, 3))
    layer = inception_module(inputs, 64, 96, 128, 16, 32, 32)
    layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
    model = tf.keras.Model(inputs=inputs, outputs=layer)
    tf.keras.utils.plot_model(model, show_shapes=True)
	"""
	# 1x1 conv
	conv1 = tf.keras.layers.Conv2D(f1, (1,1), padding='same', activation=act)(layer_in)
	# 3x3 conv
	conv3 = tf.keras.layers.Conv2D(f2_in, (1,1), padding='same', activation=act)(layer_in)
	conv3 = tf.keras.layers.Conv2D(f2_out, (3,1), padding='same', activation=act)(conv3)
	# 5x5 conv
	conv5 = tf.keras.layers.Conv2D(f3_in, (1,1), padding='same', activation=act)(layer_in)
	conv5 = tf.keras.layers.Conv2D(f3_out, (5,1), padding='same', activation=act)(conv5)
	# 3x3 max pooling
	pool = tf.keras.layers.MaxPool2D((3,1), strides=(1,1), padding='same')(layer_in)
	pool = tf.keras.layers.Conv2D(f4_out, (1,1), padding='same', activation=act)(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out
	
def inception_module2(layer_in, f, sizes,act):
	"""
	layer_in: input data shape
    f1: # of filters in 1x1 conv
	f2_in, f2_out: # of filters in the consecutive 1x1 conv and 3x3 conv layers
	f3_in, f3_out: # of filters in the consecutive 1x1 conv and 5x5 conv layers
	f4_out: # of filters in the conv layer after MaxPool layer

	Example:
    inputs = tf.keras.Input(shape=(256, 256, 3))
    layer = inception_module(inputs, 64, 96, 128, 16, 32, 32)
    layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
    model = tf.keras.Model(inputs=inputs, outputs=layer)
    tf.keras.utils.plot_model(model, show_shapes=True)
	"""
	# 1x1 conv
	conv1 = tf.keras.layers.Conv2D(f, (sizes[0],1), padding='same', activation=act)(layer_in)
	# 3x3 conv
	conv3 = tf.keras.layers.Conv2D(f, (1,1), padding='same', activation=act)(layer_in)
	conv3 = tf.keras.layers.Conv2D(f, (sizes[1],1), padding='same', activation=act)(conv3)
	# 5x5 conv
	conv5 = tf.keras.layers.Conv2D(f, (1,1), padding='same', activation=act)(layer_in)
	conv5 = tf.keras.layers.Conv2D(f, (sizes[2],1), padding='same', activation=act)(conv5)
	# 3x3 max pooling
	pool = tf.keras.layers.MaxPool2D((3,1), strides=(1,1), padding='same')(layer_in)
	pool = tf.keras.layers.Conv2D(f, (1,1), padding='same', activation=act)(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import add
from tensorflow.keras.utils import plot_model

def residual_module(layer_in, n_filters):
	"""
	visible = Input(shape=(256, 256, 3))
	layer = residual_module(visible, 64)
	model = Model(inputs=visible, outputs=layer)
	model.summary()
	plot_model(model, show_shapes=True)
	"""
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = Conv2D(n_filters, (3,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = Conv2D(n_filters, (3,1), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])
	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out
