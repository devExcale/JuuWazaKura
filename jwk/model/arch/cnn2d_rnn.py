import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, RNN, GRUCell, Dense, TimeDistributed


class CNN2DRNN(Model):

	def __init__(self, frame_size: tuple[int, int], num_throws: int, *args, **kwargs):
		"""

		:param frame_size:
		:param num_throws:
		"""
		super().__init__(*args, **kwargs)

		self.frame_size = frame_size
		""" Size of each frame in the video (H, W). """

		self.num_throws = num_throws
		""" Number of throw classes. """

		self.__model__()

		return

	def __model__(self):
		"""
		Build the CNN2DRNN model.
		"""

		input_shape = (self.frame_size[0], self.frame_size[1], 3)

		self.conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)
		self.conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')
		self.pool1 = MaxPooling2D((2, 2))

		self.conv21 = Conv2D(64, (3, 3), activation='relu', padding='same')
		self.conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')
		self.pool2 = MaxPooling2D((2, 2))

		self.conv31 = Conv2D(128, (3, 3), activation='relu', padding='same')
		self.conv32 = Conv2D(128, (3, 3), activation='relu', padding='same')
		self.pool3 = GlobalAveragePooling2D()

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

		self.rnn1 = RNN(GRUCell(64), return_sequences=True)
		self.rnn2 = RNN(GRUCell(64), return_sequences=True)
		self.rnn3 = RNN(GRUCell(64), return_sequences=False)

		self.dense1 = Dense(64, activation='relu')
		self.dense2 = Dense(32, activation='relu')

		self.dense_throw = Dense(self.num_throws, activation='softmax', name='throw')
		self.dense_tori = Dense(2, activation='softmax', name='tori')

		return

	@tf.function
	def call(self, inputs: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
		"""
		Forward pass through the CNN2DRNN model.

		:param inputs: Input tensor of shape (T, H, W, C)
		:param training: Whether the model is in training mode.
		:return: Tuple of outputs (throw_output, tori_output)
		"""

		# inputs shape: (batch_size, timesteps, H, W, C)

		# To apply cnn_feature_extractor to each time slice using tf.map_fn,
		# we typically want the dimension we're mapping over to be the first one.
		# So, transpose inputs from (batch_size, timesteps, H, W, C)
		# to (timesteps, batch_size, H, W, C).
		x_permuted = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

		# Now, apply the cnn_feature_extractor to each time step.
		# `tf.map_fn` will iterate over the first dimension of `x_permuted` (timesteps).
		# Each `frame_batch_slice` passed to the lambda will have shape (batch_size, H, W, C).
		# The output of `self.cnn_feature_extractor` will be (batch_size, cnn_output_features).
		processed_frames = tf.map_fn(
			lambda frame_batch_slice: self.cnn(frame_batch_slice, training=training),
			x_permuted
		)
		# `processed_frames` will have shape (timesteps, batch_size, cnn_output_features)

		# Transpose back to the conventional (batch_size, timesteps, cnn_output_features) for RNN layers
		x_rnn_input = tf.transpose(processed_frames, perm=[1, 0, 2])

		# --- Pass through RNN and Dense layers ---
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

	def train_step(self, data):
		x, y_true_dict = data

		with tf.GradientTape() as tape:
			y_pred_dict = self(x, training=True)

			loss = self.compiled_loss(
				y_true_dict, y_pred_dict, regularization_losses=self.losses
			)

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update metrics
		self.compiled_metrics.update_state(y_true_dict, y_pred_dict)

		return {m.name: m.result() for m in self.metrics}
