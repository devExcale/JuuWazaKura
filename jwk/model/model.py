import logging
import os
from typing import Callable

import psutil
import tensorflow as tf
from tensorflow.keras import backend, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Lambda, Concatenate, Reshape, GRU, \
	MaxPooling2D, Conv2D, LSTM, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Recall, Precision
from tensorflow.keras.optimizers import Adam

from ..dataset.ds_generator import DatasetBatchGenerator
from ..dataset.ds_handler import DatasetHandler
from ..utils import MyEnv, get_logger

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())

backend.set_floatx('float32')


class JwkModel:
	"""
	Class to handle the model.
	"""

	def __init__(self, model_type: str):
		"""
		Initialize the model handler object.

		:param model_type: Code of the model to enable.
		"""

		self.dataset: DatasetHandler = DatasetHandler()
		""" Dataset handler object. """

		self.target_size: tuple[int, int] = (112, 112)
		""" Target size of the input frames. """

		self.target_frames: int = 24
		""" Number of frames to use as input. """

		self.frame_window: int = 5
		""" Number of frames to use as a sliding window. """

		self.segments_per_batch: int = MyEnv.segments_per_batch

		self.model: Model | None = None
		""" Model object. """

		self.optimizer = Adam(learning_rate=0.001)
		self.loss_fn = CategoricalCrossentropy()
		self.accuracy_throw = CategoricalAccuracy(name='acc')
		self.accuracy_tori = CategoricalAccuracy(name='acc')
		self.recall = Recall(name='rec')
		self.precision = Precision(name='prc')

		enable_model: Callable[['JwkModel'], None] | None
		enable_model = self.models.get(model_type, None)

		if not enable_model:
			raise ValueError(f'No model {model_type} found.')

		# Load entire dataset (csv records)
		self.dataset.load_all(MyEnv.dataset_source, set(MyEnv.dataset_include), set(MyEnv.dataset_exclude))
		self.dataset.finalize()

		self.batch_generator: DatasetBatchGenerator | None = None

		enable_model(self)

		return

	def fit_model(self, epochs=10):

		if self.model is None:
			raise ValueError('No model to train!')

		# Early Stopping: Stop training if validation loss doesn't improve for 15 epochs
		early_stopping = EarlyStopping(
			monitor='tori_loss',  # Or 'val_throw_loss', 'val_tori_loss', or 'val_throw_accuracy' etc.
			patience=20,  # Number of epochs with no improvement after which training will be stopped.
			mode='min',  # 'min' for loss, 'max' for accuracy
			restore_best_weights=True  # Restore model weights from the epoch with the best monitored value
		)

		# Learning Rate Scheduler: Reduce LR if validation loss plateaus
		lr_scheduler = ReduceLROnPlateau(
			monitor='tori_loss',  # Monitor validation loss
			factor=0.5,  # Reduce learning rate by half
			patience=5,  # Reduce after 5 epochs with no improvement (often less than early stopping patience)
			min_lr=0.00001,  # Don't let the learning rate fall below this
			verbose=1  # Print messages when the learning rate is reduced
		)

		callbacks = [early_stopping, lr_scheduler]

		self.model.fit(
			self.batch_generator,
			epochs=epochs,
			workers=8,
			use_multiprocessing=True,
			max_queue_size=12,
			callbacks=callbacks,
		)

		return

	def enable_3dcnn(self):

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		# Create dataset loader
		self.batch_generator = DatasetBatchGenerator(
			dataset=self.dataset,
			segments_per_batch=self.segments_per_batch,
			input_size=self.target_size,
			fixed_frames=self.target_frames,
		)

		# Define shapes
		input_shape = (self.target_frames, *self.target_size, 3)
		num_throws = len(self.dataset.throw_classes)
		num_tori = len(self.dataset.tori_classes)

		input_layer = Input(batch_shape=input_shape)

		layer = Conv3D(
			filters=32,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='relu',
		)(input_layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		layer = Conv3D(
			filters=64,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='relu',
		)(layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		layer = Conv3D(
			filters=128,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='relu',
		)(layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		layer = Flatten()(layer)

		layer = Dense(256, activation='relu')(layer)

		layer = Dense(128, activation='relu')(layer)

		# Define output layers
		layer_output_throw = Dense(num_throws, name='throw', activation='softmax')(layer)
		layer_output_tori = Dense(num_tori, name='tori', activation='softmax')(layer)

		# Compile the model
		self.model = Model(inputs=input_layer, outputs=[layer_output_throw, layer_output_tori])
		self.model.compile(
			optimizer='adam',
			metrics={
				'throw': [self.accuracy_throw],
				'tori': [self.accuracy_tori],
			},
			loss={
				'throw': self.loss_fn,
				'tori': self.loss_fn,
			},
			loss_weights={
				'throw': 0.8,
				'tori': 0.2,
			},
		)

		return

	def enable_conv3d_gru(self):

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		self.batch_generator = DatasetBatchGenerator(
			dataset=self.dataset,
			segments_per_batch=self.segments_per_batch,
			input_size=self.target_size,
			window_frames=self.frame_window,
		)

		num_throws = len(self.dataset.throw_classes)
		num_tori = len(self.dataset.tori_classes)

		batch_shape = (self.batch_generator.batch_size, None, *self.target_size, 3)

		input_layer = Input(batch_shape=batch_shape)

		layer = Conv3D(
			filters=32,
			kernel_size=(self.frame_window, 5, 5),
			strides=(1, 1, 1),
			padding='same',
			activation='elu',
		)(input_layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		layer = Conv3D(
			filters=64,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='elu',
		)(layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		layer = Conv3D(
			filters=128,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='elu',
		)(layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		# Compute the flattened feature size: conv3d filters * downsampled frame size
		dim_flat_features = 128 * (self.target_size[0] // 8) * (self.target_size[1] // 8)

		layer = Reshape((-1, dim_flat_features))(layer)

		# layer = LSTM(units=64)(layer)
		layer = GRU(units=128)(layer)

		layer = Dense(256, activation='elu')(layer)

		layer = Dense(128, activation='elu')(layer)

		# Define output layers
		layer_output_throw = Dense(num_throws, name='throw', activation='softmax')(layer)
		layer_output_tori = Dense(num_tori, name='tori', activation='softmax')(layer)

		self.model = Model(inputs=input_layer, outputs=[layer_output_throw, layer_output_tori])
		self.model.compile(
			optimizer=self.optimizer,
			metrics={
				'throw': [self.accuracy_throw],
				'tori': [self.accuracy_tori],
			},
			loss={
				'throw': self.loss_fn,
				'tori': self.loss_fn,
			},
			loss_weights={
				'throw': 0.8,
				'tori': 0.2,
			},
		)

		return

	def enable_2dcnn_lstm(self):
		"""
		Configures a 2D CNN + LSTM model for variable-length sequences in batches.
		Applies 2D CNNs per frame, then processes the sequence with an LSTM.
		"""

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		self.batch_generator = DatasetBatchGenerator(
			dataset=self.dataset,
			segments_per_batch=16,
			input_size=self.target_size,
			window_frames=5,  # Not necessary, temp used for internal handling
		)

		num_throws = len(self.dataset.throw_classes)
		num_tori = len(self.dataset.tori_classes)

		batch_shape = (None, None, *self.target_size, 3)
		input_layer = tf.keras.Input(batch_shape=batch_shape)

		cnn_block = tf.keras.Sequential([
			Conv2D(32, (3, 3), activation='relu', padding='same'),
			MaxPooling2D((2, 2)),

			Conv2D(64, (3, 3), activation='relu', padding='same'),
			MaxPooling2D((2, 2)),

			Conv2D(128, (3, 3), activation='relu', padding='same'),

			# Use GlobalAveragePooling2D to flatten spatial dimensions
			tf.keras.layers.GlobalAveragePooling2D(),
			# Or Flatten() if you prefer, potentially followed by a small Dense layer
			# tf.keras.layers.Flatten(),
			# tf.keras.layers.Dense(64, activation='relu')
		])

		features_sequence = tf.keras.layers.TimeDistributed(cnn_block)(input_layer)

		layer = LSTM(units=128)(features_sequence)

		layer = Dense(128, activation='relu')(layer)
		layer = Dense(64, activation='relu')(layer)

		layer_output_throw_logits = Dense(num_throws, activation='linear')(layer)
		layer_output_tori_logits = Dense(num_tori, activation='linear')(layer)

		layer_output_throw = Softmax(name='throw')(layer_output_throw_logits)
		layer_output_tori = Softmax(name='tori')(layer_output_tori_logits)

		self.model = tf.keras.Model(inputs=input_layer, outputs=[layer_output_throw, layer_output_tori])
		self.model.compile(
			optimizer=self.optimizer,
			loss={
				'throw': self.loss_fn,
				'tori': self.loss_fn
			},
			metrics={
				'throw': [self.accuracy_throw],
				'tori': [self.accuracy_tori],
			},
			loss_weights={
				'throw': 0.8,
				'tori': 0.2,
			},
		)

		return

	def _layer_concatenate_softmax(self, layer_output: tf.Tensor) -> tf.Tensor:
		"""
		Apply softmax to the different outputs of the model.

		:param layer_output: Output of the model.
		:return: Softmax output of the model.
		"""

		# Get the number of classes
		num_throws = len(self.dataset.throw_classes)

		# Separate the output into two parts
		out_throw = Lambda(lambda lyr: lyr[:, :num_throws])(layer_output)
		out_tori = Lambda(lambda lyr: lyr[:, num_throws:])(layer_output)

		# Apply softmax activation to each part
		softmax_throw = Lambda(lambda lyr: tf.nn.softmax(lyr))(out_throw)
		softmax_tori = Lambda(lambda lyr: tf.nn.softmax(lyr))(out_tori)

		return Concatenate()([softmax_throw, softmax_tori])

	def save_weights(self, filepath):
		"""
		Save the model weights to a file.

		:param filepath:
		:return:
		"""
		self.model.save_weights(filepath)

		return

	def load_weights(self, filepath):
		self.model.load_weights(filepath)

		return

	models = {
		'3dcnn': enable_3dcnn,
		'3dcnn_gru': enable_conv3d_gru,
		'2dcnn_lstm': enable_2dcnn_lstm,
	}


def get_memory_usage():
	"""Returns the current memory usage (in MB)."""

	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()

	return mem_info.rss / (1024 * 1024)
