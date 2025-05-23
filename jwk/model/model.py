import logging
import os
from typing import Callable

# noinspection PyPackageRequirements
import absl.logging
import click
import keras.mixed_precision
import psutil
import tensorflow as tf
from keras import Model, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Lambda, Concatenate, Reshape, MaxPooling2D, Conv2D, \
	LSTMCell, RNN, Softmax, MultiHeadAttention, Add, GRUCell
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, Recall, Precision
from tensorflow_addons.optimizers import AdamW

from .arch.recompute_grad_sequential import RecomputeGradSequential
from ..dataset.ds_generator import DatasetBatchGenerator
from ..dataset.ds_handler import DatasetHandler, normalize_label
from ..utils import MyEnv, get_logger

absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())

keras.mixed_precision.set_global_policy('float32')


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

		self.optimizer = AdamW(learning_rate=0.0001, weight_decay=0.0001)
		self.loss_fn = CategoricalCrossentropy()
		self.accuracy_throw = CategoricalAccuracy(name='acc')
		self.accuracy_tori = CategoricalAccuracy(name='acc')
		self.recall = Recall(name='rec')
		self.precision = Precision(name='prc')

		enable_model: Callable[['JwkModel'], None] | None
		enable_model = self.models.get(model_type, None)

		if not enable_model:
			raise ValueError(f'No model {model_type} found.')

		keeplist = [
			"seoi_nage",
			"uchi_mata",
			"sode_tsurikomi_goshi",
			"o_uchi_gari",
			"sumi_gaeshi",
			# "kata_guruma",
			# "tai_otoshi",
			# "ko_soto_gari",
			# "morote_seoi_nage",
			# "tani_otoshi",
		]

		# Load entire dataset (csv records)
		self.dataset.load_all(MyEnv.dataset_source, set(MyEnv.dataset_include), set(MyEnv.dataset_exclude))

		df = self.dataset.df
		df.drop(df[
					~df['throw']
				.map(normalize_label)
				.map(lambda t: self.dataset.throw_from_alias.get(t, t))
				.map(lambda t: self.dataset.throw_to_group.get(t, t))
				.isin(keeplist)
				].index, inplace=True)
		df.reset_index(drop=True, inplace=True)
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

		checkpoint = ModelCheckpoint(
			filepath='checkpoints/model-{epoch:02d}-l{throw_loss:.2f}-a{throw_acc:.2f}.tf',
			monitor='throw_acc',
			save_weights_only=True,
			save_best_only=True,
			mode='max',
			save_freq='epoch',
		)

		callbacks = [early_stopping, lr_scheduler, checkpoint]

		self.model.fit(
			self.batch_generator,
			epochs=epochs,
			workers=8,
			use_multiprocessing=True,
			max_queue_size=6,
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
			halve_fps=True,
		)

		num_throws = len(self.dataset.throw_classes)
		num_tori = len(self.dataset.tori_classes)

		click.echo(f'Number of throws: {num_throws}')

		batch_shape = (self.batch_generator.batch_size, None, *self.target_size, 3)

		input_layer = Input(batch_shape=batch_shape)

		layer = Conv3D(
			filters=16,
			kernel_size=(self.frame_window, 5, 5),
			strides=(1, 1, 1),
			padding='same',
			activation='elu',
		)(input_layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		layer = Conv3D(
			filters=32,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='elu',
		)(layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		layer = Conv3D(
			filters=64,
			kernel_size=(3, 3, 3),
			strides=(1, 1, 1),
			padding='same',
			activation='elu',
		)(layer)

		layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

		# Compute the flattened feature size: conv3d filters * downsampled frame size
		dim_flat_features = 64 * (self.target_size[0] // 8) * (self.target_size[1] // 8)

		layer = Reshape((-1, dim_flat_features))(layer)

		# layer = LSTM(units=64)(layer)

		# layer = GRU(units=128)(layer)
		layer = RNN(GRUCell(units=128))(layer)

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
			segments_per_batch=self.segments_per_batch,
			input_size=self.target_size,
			window_frames=5,  # Not necessary, temp used for internal handling
			halve_fps=True,
		)

		num_throws = len(self.dataset.throw_classes)
		num_tori = len(self.dataset.tori_classes)

		batch_shape = (self.batch_generator.batch_size, None, *self.target_size, 3)
		input_layer = tf.keras.Input(batch_shape=batch_shape)

		cnn_block = RecomputeGradSequential([
			# Conv2D(32, (11, 11), activation='relu', padding='same'),
			# MaxPooling2D((2, 2)),

			Conv2D(32, (7, 7), activation='relu', padding='same'),
			MaxPooling2D((2, 2)),

			Conv2D(64, (5, 5), activation='relu', padding='same'),
			MaxPooling2D((2, 2)),

			Conv2D(128, (3, 3), activation='relu', padding='same'),

			# Use GlobalAveragePooling2D to flatten spatial dimensions
			tf.keras.layers.GlobalAveragePooling2D(),
			# Or Flatten() if you prefer, potentially followed by a small Dense layer
			# tf.keras.layers.Flatten(),
			# tf.keras.layers.Dense(64, activation='relu')
		])

		features_sequence = tf.keras.layers.TimeDistributed(cnn_block)(input_layer)

		layer = RNN(LSTMCell(units=64), return_sequences=True)(features_sequence)
		layer = RNN(LSTMCell(units=64), return_sequences=True)(layer)
		layer = RNN(LSTMCell(units=64))(layer)

		layer = Dense(128, activation='relu')(layer)
		layer = Dense(64, activation='relu')(layer)
		layer = Dense(32, activation='relu')(layer)

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

	def enable_2dcnn_transformer(self):
		"""
		Configures a 2D CNN + Transformer Encoder model for FIXED-length sequences IN BATCHES.
		(Masking layer REMOVED as requested - NOT suitable for padded variable length batches).
		"""
		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		# Configure batch generator.
		# NOTE: For THIS model version (no Masking), the generator MUST
		# yield batches where ALL sequences have the SAME fixed length.
		self.batch_generator = DatasetBatchGenerator(
			dataset=self.dataset,
			segments_per_batch=self.segments_per_batch,  # Use batch size > 1
			input_size=self.target_size,  # (224, 224) example
			window_frames=5,  # Not necessary, temp used for internal handling
		)

		num_throws = len(self.dataset.throw_classes)
		num_tori = len(self.dataset.tori_classes)

		# Input shape: (batch_size, fixed_sequence_length, height, width, channels)
		# Set sequence length to None for flexibility between DIFFERENT batches
		# but WITHIN a batch, sequences must be same length for THIS version.
		# If you use padding, the second None becomes the max_sequence_length_in_batch.
		batch_shape = (self.batch_generator.batch_size, None, *self.target_size, 3)
		input_layer = tf.keras.Input(batch_shape=batch_shape)

		transformer_units = 64
		num_heads = 4
		ff_dim = 64

		# --- 2D CNN Block (applied per frame using TimeDistributed) ---
		# Re-using the sequence of layers we evaluated: 7x7:32, 5x5:64, 3x3:128 + Pool + GlobalAvgPool
		cnn_block = tf.keras.Sequential([
			tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
			tf.keras.layers.MaxPooling2D((2, 2)),

			tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
			tf.keras.layers.MaxPooling2D((2, 2)),

			tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
			tf.keras.layers.MaxPooling2D((2, 2)),

			tf.keras.layers.GlobalAveragePooling2D(),  # Output features per frame: 128

			Dense(64, activation='relu'),
		])

		# Apply the 2D CNN block to each time step
		# features_sequence shape: (batch_size, sequence_length, 128)
		features_sequence = tf.keras.layers.TimeDistributed(cnn_block)(input_layer)

		# --- Positional Encoding ---
		# Add positional encoding to the features sequence.
		# This requires knowing the sequence length (max_sequence_length_in_batch).
		# This approach assumes the generator yields batches of uniform sequence length.
		# If using padding, this length is dynamic per batch, and you MUST
		# use a Masking layer or mask_zero on the *first* layer receiving padded input.

		# Get the sequence length of the current batch
		max_sequence_length_in_batch = tf.shape(features_sequence)[1]
		position_ids = tf.range(start=0, limit=max_sequence_length_in_batch, delta=1)

		# Use an Embedding layer for learned positional embeddings
		# input_dim needs to be large enough for max sequence length across ALL batches
		position_embedding_layer = tf.keras.layers.Embedding(
			input_dim=1000,  # Example max possible sequence length (adjust as needed)
			output_dim=transformer_units  # Should match feature dimension (128 here)
		)
		position_embeddings = position_embedding_layer(position_ids)

		# Add positional embeddings to the feature sequence
		# If features_sequence comes from TimeDistributed(..., padding='valid'),
		# or temporal strides are not 1, sequence length might be less than input_layer's T.
		# Ensure shapes align for addition: (batch_size, seq_len, features) + (seq_len, features) via broadcasting
		input_to_transformer = features_sequence + position_embeddings

		# --- Transformer Encoder Block (Masking layer REMOVED) ---
		x = input_to_transformer
		# NOTE: THIS BLOCK WILL PROCESS PADDING IF YOUR GENERATOR USES PADDING
		# AND YOU DON'T HAVE A MASKING LAYER OR mask_zero=True!
		for _ in range(2):
			# Layer Normalization before Attention
			x_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

			# Multi-Head Self-Attention - WILL ATTEND TO PADDING WITHOUT MASK!
			attn_output = MultiHeadAttention(
				num_heads=num_heads,
				key_dim=transformer_units // num_heads,
				dropout=0.1
			)(x_norm1, x_norm1)  # No attention_mask passed here!

			# Add attention output to the input (Residual connection) and apply Dropout
			x_add1 = Add()([attn_output, x])
			x_drop1 = tf.keras.layers.Dropout(0.1)(x_add1)

			# Layer Normalization before Feed-Forward
			x_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_drop1)

			# Feed-Forward Network
			ff_output = tf.keras.layers.Dense(ff_dim, activation='relu')(x_norm2)
			ff_output = tf.keras.layers.Dense(transformer_units)(ff_output)

			# Add FF output to the input (Residual connection) and apply Dropout
			x_add2 = Add()([ff_output, x_drop1])
			x = tf.keras.layers.Dropout(0.1)(x_add2)

		transformer_output_sequence = x  # Shape: (batch_size, sequence_length, transformer_units)

		# --- Pooling After Transformer ---
		# Collapse the temporal dimension
		pooled_features = tf.keras.layers.GlobalAveragePooling1D()(
			transformer_output_sequence)  # Output shape: (batch_size, transformer_units)

		# --- Final Dense layers for classification ---
		layer = tf.keras.layers.Dense(64, activation='relu')(pooled_features)
		layer = tf.keras.layers.Dense(32, activation='relu')(layer)

		# --- Define Separate Output Layers ---
		layer_output_throw_logits = tf.keras.layers.Dense(num_throws, name='throw_logits', activation='linear')(layer)
		layer_output_tori_logits = tf.keras.layers.Dense(num_tori, name='tori_logits', activation='linear')(layer)

		# Apply Softmax activation (renamed for compile)
		layer_output_throw = tf.keras.layers.Softmax(name='throw')(layer_output_throw_logits)
		layer_output_tori = tf.keras.layers.Softmax(name='tori')(layer_output_tori_logits)

		# --- Define the Model with Multiple Outputs ---
		self.model = tf.keras.Model(inputs=input_layer, outputs=[layer_output_throw, layer_output_tori])

		# --- Compile the Model ---
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
		'2dcnn_transformer': enable_2dcnn_transformer,
	}


def get_memory_usage():
	"""Returns the current memory usage (in MB)."""

	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()

	return mem_info.rss / (1024 * 1024)
