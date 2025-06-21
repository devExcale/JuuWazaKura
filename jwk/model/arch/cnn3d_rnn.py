import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, RNN, GRUCell, Dense, GlobalAveragePooling2D, \
	TimeDistributed


class CNN3DRNN(Model):

	def __init__(self, frame_size: tuple[int, int], num_throws: int, window_size: int, *args, **kwargs):
		"""
		:param frame_size: Tuple (H, W) representing the height and width of each frame.
		:param num_throws: Number of throw classes.
		:param window_size: Number of consecutive frames in each 3D CNN block.
		"""
		super().__init__(*args, **kwargs)

		self.frame_size = frame_size
		self.num_throws = num_throws
		self.window_size = window_size

		self.__model__()

		return

	def __model__(self):
		"""
		Build the CNN3DRNN model.
		"""
		input_shape = (None, self.frame_size[0], self.frame_size[1], 3)

		# 3D CNN layers
		self.conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape)
		self.conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')
		self.pool1 = MaxPooling3D((1, 2, 2))

		self.conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')
		self.conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')
		self.pool2 = MaxPooling3D((1, 2, 2))

		self.conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')
		self.conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')
		self.pool3 = MaxPooling3D((1, 2, 2))

		self.cnn = Sequential([
			self.conv11,
			self.conv12,
			self.pool1,
			self.conv21,
			self.conv22,
			self.pool2,
			self.conv31,
			self.conv32,
			self.pool3
		])

		# Apply 2d pooling for each timestep
		self.td_gap = TimeDistributed(GlobalAveragePooling2D())

		# RNN layers
		self.rnn1 = RNN(GRUCell(64), return_sequences=True)
		self.rnn2 = RNN(GRUCell(64), return_sequences=True)
		self.rnn3 = RNN(GRUCell(64), return_sequences=False)

		# Dense layers
		self.dense1 = Dense(64, activation='relu')
		self.dense2 = Dense(32, activation='relu')

		self.dense_throw = Dense(self.num_throws, activation='softmax', name='throw')
		self.dense_tori = Dense(2, activation='softmax', name='tori')

	@tf.function
	def call(self, inputs: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
		"""
		Forward pass through the CNN3DRNN model.

		:param inputs: Input tensor of shape (batch_size, timesteps, H, W, C)
		:param training: Whether the model is in training mode.
		:return: Dictionary of outputs (throw_output, tori_output)
		"""

		# Pass through 3D CNN layers
		x = self.cnn(inputs, training=training)

		# Reduce dim to (batch_size, timesteps, features)
		x = self.td_gap(x, training=training)

		# Pass through RNN and Dense layers
		x = self.rnn1(x, training=training)
		x = self.rnn2(x, training=training)
		x = self.rnn3(x, training=training)

		# Pass RNN features through MLP
		x = self.dense1(x, training=training)
		x = self.dense2(x, training=training)

		throw_output = self.dense_throw(x, training=training)
		tori_output = self.dense_tori(x, training=training)

		return {
			'throw': throw_output,
			'tori': tori_output
		}
