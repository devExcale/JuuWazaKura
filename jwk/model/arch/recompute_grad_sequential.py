import tensorflow as tf
from tensorflow.keras.layers import Layer


class RecomputeGradSequential(Layer):
	"""
	Convenience class to recompute gradients for a sequential block of layers during training.
	"""

	def __init__(self, layers_list: list[Layer], **kwargs):
		super().__init__(**kwargs)

		# Define sequential layer
		self.sequential = tf.keras.Sequential(layers_list)

		return

	@tf.recompute_grad
	def call(self, inputs, training=None, mask=None):
		return self.sequential(inputs, training=training, mask=mask)
