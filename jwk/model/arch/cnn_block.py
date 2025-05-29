from keras.layers import Layer, Conv2D, MaxPooling2D, GlobalAveragePooling2D


class CNNBlock(Layer):
	def __init__(self, name="cnn_block", **kwargs):
		super().__init__(name=name, **kwargs)

		self.conv1 = Conv2D(32, (7, 7), activation='relu', padding='same')
		self.pool1 = MaxPooling2D((2, 2))

		self.conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')
		self.pool2 = MaxPooling2D((2, 2))

		self.conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')
		self.pool3 = GlobalAveragePooling2D()

		return

	def call(self, inputs, **kwargs):
		"""
		Forward pass through the CNN block.
		"""

		x = self.conv1(inputs)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.pool3(x)

		return x
