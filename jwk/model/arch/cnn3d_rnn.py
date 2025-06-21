import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, RNN, GRUCell, Dense


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

	def __model__(self):
		"""
		Build the CNN3DRNN model.
		"""
		input_shape = (self.window_size, self.frame_size[0], self.frame_size[1], 3)

		# 3D CNN layers
		self.conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape)
		self.conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')
		self.pool1 = MaxPooling3D((1, 2, 2))  # Pooling only in spatial dimensions

		self.conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')
		self.conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')
		self.pool2 = MaxPooling3D((1, 2, 2))

		self.conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')
		self.conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')
		self.pool3 = GlobalAveragePooling3D()

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
		# Create sliding windows of frames
		timesteps = inputs.shape[1]  # Number of frames in the video
		windowed_inputs = tf.image.extract_patches(
			images=inputs,
			sizes=[1, self.window_size, 1, 1, 1],
			strides=[1, 1, 1, 1, 1],
			rates=[1, 1, 1, 1, 1],
			padding='VALID'
		)
		# Reshape to (batch_size, new_timesteps, window_size, H, W, C)
		new_timesteps = timesteps - self.window_size + 1
		windowed_inputs = tf.reshape(
			windowed_inputs,
			(-1, new_timesteps, self.window_size, self.frame_size[0], self.frame_size[1], 3)
		)

		# Apply 3D CNN to each block
		x_permuted = tf.transpose(windowed_inputs, perm=[1, 0, 2, 3, 4, 5])  # (new_timesteps, batch_size, ...)
		processed_blocks = tf.map_fn(
			lambda block: self.cnn(block, training=training),
			x_permuted
		)
		# Transpose back to (batch_size, new_timesteps, cnn_output_features)
		x_rnn_input = tf.transpose(processed_blocks, perm=[1, 0, 2])

		# Pass through RNN and Dense layers
		x = self.rnn1(x_rnn_input, training=training)
		x = self.rnn2(x, training=training)
		x = self.rnn3(x, training=training)

		x = self.dense1(x)
		x = self.dense2(x)

		throw_output = self.dense_throw(x)
		tori_output = self.dense_tori(x)

		return {
			'throw': throw_output,
			'tori': tori_output
		}
