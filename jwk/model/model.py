import logging
import os
from enum import Enum, auto

import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras import backend, Model, Input
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Lambda, Concatenate, Reshape, GRU
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Recall, Precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

	def __init__(self):
		"""
		Initialize the model handler object.
		"""

		self.dataset: DatasetHandler = DatasetHandler()
		""" Dataset handler object. """

		self.target_size: tuple[int, int] = (224, 224)
		""" Target size of the input frames. """

		self.target_frames: int = 24
		""" Number of frames to use as input. """

		self.frame_window: int = 6
		""" Number of frames to use as a sliding window. """

		self.segments_per_batch: int = 8

		self.model: Model | None = None
		""" Model object. """

		self.optimizer = Adam(learning_rate=0.001)
		self.loss_fn = CategoricalCrossentropy()
		self.accuracy_throw = CategoricalAccuracy(name='accuracy')
		self.accuracy_tori = CategoricalAccuracy(name='accuracy')
		self.recall = Recall(name='recall')
		self.precision = Precision(name='precision')

		# Load entire dataset (csv records)
		self.dataset.load_all(MyEnv.dataset_source, set(MyEnv.dataset_include), set(MyEnv.dataset_exclude))
		self.dataset.finalize()

		self.batch_generator: DatasetBatchGenerator | None = None

		return

	def enable_3dcnn(self):

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		input_shape = (self.target_frames, *self.target_size, 3)
		num_throws = len(self.dataset.throw_classes)
		num_tori = len(self.dataset.tori_classes)

		input_layer = Input(shape=input_shape)
		output_layer = Dense(num_throws + num_tori, activation='linear')

		# Define model
		base_model = Sequential([
			input_layer,
			Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
			MaxPooling3D((2, 2, 2)),
			Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
			MaxPooling3D((2, 2, 2)),
			Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
			MaxPooling3D((2, 2, 2)),
			Flatten(),
			Dense(256, activation='relu'),
			output_layer,
		])

		base_output = output_layer.output

		# Apply softmax activation to separate outputs
		model_output = self._layer_concatenate_softmax(base_output)

		# Compile the model
		self.model = Model(inputs=input_layer, outputs=model_output)
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		self.batch_generator = DatasetBatchGenerator(
			dataset=self.dataset,
			segments_per_batch=self.segments_per_batch,
			input_size=self.target_size,
			fixed_frames=self.target_frames,
		)

		return

	def enable_3dcnn_rnn(self):

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		self.batch_generator = DatasetBatchGenerator(
			dataset=self.dataset,
			segments_per_batch=1,
			input_size=self.target_size,
			window_frames=self.frame_window,
		)

		num_throws = len(self.dataset.throw_classes)
		num_tori = len(self.dataset.tori_classes)
		num_classes = num_throws + num_tori

		batch_shape = (self.batch_generator.batch_size, None, *self.target_size, 3)

		input_layer = Input(batch_shape=batch_shape)

		layer = Conv3D(
			filters=16,
			kernel_size=(self.frame_window, 5, 5),
			strides=(self.frame_window // 2, 1, 1),
			padding='same',
			activation='relu',
		)(input_layer)

		layer = MaxPooling3D(pool_size=(1, 2, 2))(layer)

		layer = Conv3D(
			filters=32,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='relu',
		)(layer)

		layer = MaxPooling3D(pool_size=(1, 2, 2))(layer)

		# Compute the flattened feature size: conv3d filters * downsampled frame size
		dim_flat_features = 32 * (self.target_size[0] // 4) * (self.target_size[1] // 4)

		layer = Reshape((-1, dim_flat_features))(layer)

		# layer = LSTM(units=64)(layer)
		layer = GRU(units=32)(layer)

		layer = Dense(32, activation='relu')(layer)

		output_layer = Dense(num_classes, activation='linear')(layer)

		# Apply softmax activation to separate outputs
		model_output = self._layer_concatenate_softmax(output_layer)

		self.model = Model(inputs=input_layer, outputs=model_output)
		self.model.compile(
			optimizer=self.optimizer,
			loss=self.loss_fn,
			metrics=[self.accuracy_tori, self.accuracy_throw, self.recall, self.precision],
		)

		return

	def enable_new(self):

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
			filters=16,
			kernel_size=(self.frame_window, 5, 5),
			strides=(self.frame_window // 3, 1, 1),
			padding='same',
			activation='relu',
		)(input_layer)

		layer = MaxPooling3D(pool_size=(1, 2, 2))(layer)

		layer = Conv3D(
			filters=32,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='relu',
		)(layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		layer = Conv3D(
			filters=64,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='relu',
		)(layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		# Compute the flattened feature size: conv3d filters * downsampled frame size
		dim_flat_features = 64 * (self.target_size[0] // 8) * (self.target_size[1] // 8)

		layer = Reshape((-1, dim_flat_features))(layer)

		# layer = LSTM(units=64)(layer)
		layer = GRU(units=32)(layer)

		layer = Dense(64, activation='relu')(layer)

		layer = Dense(32, activation='relu')(layer)

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

	def fit_model(self, epochs=10):

		self.enable_new()

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

	@tf.function
	def _train_step_rnn(self, batch_data: np.ndarray, batch_labels: np.ndarray) -> dict[str, tf.Tensor]:
		# Get the number of throws for slicing
		# Need to get this from the dataset handler, assuming it's accessible
		num_throws = len(self.dataset.throw_classes)

		with tf.GradientTape() as tape:
			# Forward pass
			prediction = self.model(batch_data, training=True)

			# Calculate the total loss
			loss = self.loss_fn(batch_labels, prediction)

		# Calculate gradients based on the batch loss
		gradients = tape.gradient(loss, self.model.trainable_variables)

		# Apply gradients to the optimizer once for the entire batch
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		# Slice the labels and predictions
		label_throw = batch_labels[:, :num_throws]
		label_tori = batch_labels[:, num_throws:]
		prediction_throw = prediction[:, :num_throws]
		prediction_tori = prediction[:, num_throws:]

		# Update separate accuracy metrics
		self.accuracy_throw.update_state(label_throw, prediction_throw)
		self.accuracy_tori.update_state(label_tori, prediction_tori)

		return {
			'loss': loss,
			'accuracy_throw': self.accuracy_throw.result(),
			'accuracy_tori': self.accuracy_tori.result(),
		}

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


def get_memory_usage():
	"""Returns the current memory usage (in MB)."""

	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()

	return mem_info.rss / (1024 * 1024)


class LayerIdx(Enum):
	INPUT = 0
	""" Input layer. """

	CONV3D_1 = auto()
	""" First Conv3D layer. """

	MAXPOOL3D_1 = auto()
	""" First MaxPooling3D layer. """

	CONV3D_2 = auto()
	""" Second Conv3D layer. """

	MAXPOOL3D_2 = auto()
	""" Second MaxPooling3D layer. """

	RESHAPE = auto()
	""" Reshape layer. """

	LSTM = auto()
	""" LSTM layer. """

	DENSE = auto()
	""" Dense layer. """

	OUTPUT = auto()
	""" Output layer. """
