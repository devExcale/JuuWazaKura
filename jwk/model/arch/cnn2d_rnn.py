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

		self.conv1 = Conv2D(32, (7, 7), activation='relu', input_shape=(self.frame_size[0], self.frame_size[1], 3))
		self.pool1 = MaxPooling2D((2, 2))

		self.conv2 = Conv2D(64, (5, 5), activation='relu')
		self.pool2 = MaxPooling2D((2, 2))

		self.conv3 = Conv2D(128, (3, 3), activation='relu')
		self.pool3 = GlobalAveragePooling2D()

		self.frame_distributed = TimeDistributed(Sequential([
			self.conv1,
			self.pool1,
			self.conv2,
			self.pool2,
			self.conv3,
			self.pool3
		]))

		self.rnn1 = RNN(GRUCell(64), return_sequences=True)
		self.rnn2 = RNN(GRUCell(64), return_sequences=True)
		self.rnn3 = RNN(GRUCell(64), return_sequences=False)

		self.dense1 = Dense(64, activation='relu')
		self.dense2 = Dense(32, activation='relu')

		self.dense_throw = Dense(self.num_throws, activation='softmax', name='throw')
		self.dense_tori = Dense(2, activation='softmax', name='tori')

		return

	def call(self, inputs, *args, **kwargs) -> dict[str, tf.Tensor]:
		"""
		Forward pass through the CNN2DRNN model.

		:param inputs: Input tensor of shape (T, H, W, C)
		:return: Tuple of outputs (throw_output, tori_output)
		"""

		x = self.frame_distributed(inputs)

		x = self.rnn1(x)
		x = self.rnn2(x)
		x = self.rnn3(x)

		x = self.dense1(x)
		x = self.dense2(x)

		throw_output = self.dense_throw(x)
		tori_output = self.dense_tori(x)

		return {
			'throw': throw_output,
			'tori': tori_output
		}
