import logging
import os
from datetime import datetime
from typing import Callable

# noinspection PyPackageRequirements
import keras.mixed_precision
import psutil
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, LambdaCallback
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, Recall, Precision
from tensorflow_addons.optimizers import AdamW

from .arch.cnn2d_rnn import CNN2DRNN
from .arch.cnn3d_rnn import CNN3DRNN
from .keras_callbacks import MemoryCleanupCallback
from ..dataset.ds_generator import DatasetBatchGenerator
from ..dataset.ds_orchestrator import DatasetOrchestrator, DATASET_CUT_TRAINING, DATASET_CUT_TESTING
from ..utils import MyEnv, get_logger

# absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())

keras.mixed_precision.set_global_policy('float32')

if MyEnv.tf_dump_debug_info:
	tf.debugging.experimental.enable_dump_debug_info(
		dump_root="logs/dump",
		tensor_debug_mode="NO_TENSOR"
	)

THROW_CLASSES = {
	"seoi_nage",
	"uchi_gari",
	"uchi_mata",
	"soto_gari",
	"sumi_gaeshi",
}


class JwkModel:
	"""
	Class to handle the model.
	"""

	def __init__(self, model_type: str):
		"""
		Initialize the model handler object.

		:param model_type: Code of the model to enable.
		"""

		self.dataset: DatasetOrchestrator = DatasetOrchestrator(DATASET_CUT_TRAINING)
		""" Training dataset orchestrator object. """

		self.test_dataset: DatasetOrchestrator = DatasetOrchestrator(DATASET_CUT_TESTING)
		""" Testing dataset orchestrator object. """

		self.target_size: tuple[int, int] = (64, 64)
		""" Target size of the input frames. """

		self.frame_window: int = 5
		""" Number of frames to use as a sliding window. """

		self.segments_per_batch: int = MyEnv.segments_per_batch

		self.model: Model | None = None
		""" Model object. """

		self.optimizer = AdamW(learning_rate=0.001, weight_decay=0.0008)
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
		params = (MyEnv.dataset_source, MyEnv.livefootage_include, MyEnv.livefootage_exclude)
		self.dataset.load_all(*params)
		self.test_dataset.load_all(*params)

		# Filter dataset
		self.dataset.filter_throw(self.filter_rows)

		# Lock dataset
		self.dataset.finalize()
		self.test_dataset.finalize()

		self.batch_generator_train: DatasetBatchGenerator | None = None
		""" Batch generator for training dataset. """

		self.batch_generator_test: DatasetBatchGenerator | None = None
		""" Batch generator for testing dataset. """

		enable_model(self)

		return

	@staticmethod
	def filter_rows(throw: str) -> bool:
		"""
		Filter rows based on the throw name.

		:param throw: The throw type to filter by.
		:return: True if the row should be included, False otherwise.
		"""

		return throw in THROW_CLASSES

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

		# checkpoint = ModelCheckpoint(
		# 	filepath='checkpoints/model-{epoch:02d}-l{throw_loss:.2f}-a{throw_acc:.2f}.tf',
		# 	monitor='throw_acc',
		# 	save_weights_only=True,
		# 	save_best_only=True,
		# 	mode='max',
		# 	save_freq='epoch',
		# )

		mem_cleanup = MemoryCleanupCallback()

		tb_profile = TensorBoard(
			log_dir="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
			histogram_freq=0,
			update_freq='epoch',
		)

		epoch_mem_usage = LambdaCallback(
			on_epoch_end=print_gpu_memory_usage,
		)

		callbacks = [
			# early_stopping,
			# lr_scheduler,
			# checkpoint,
			# mem_cleanup,
			# tb_profile,
			# epoch_mem_usage,
		]

		self.model.fit(
			self.batch_generator_train,
			validation_data=self.batch_generator_test,
			epochs=epochs,
			workers=8,
			use_multiprocessing=True,
			max_queue_size=6,
			callbacks=callbacks,
		)

		test_results = self.model.evaluate(self.batch_generator_test, verbose=1)
		log.info(f"Test results: {test_results}")

		return

	def enable_cnn2drnn(self):
		"""
		Configures a 2D CNN + LSTM model for variable-length sequences in batches.
		Applies 2D CNNs per frame, then processes the sequence with an LSTM.
		"""

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		self.batch_generator_train = DatasetBatchGenerator(
			dataset=self.dataset,
			frame_size=self.target_size,
			frame_stride=2,
			frame_stride_augment=True,
		)

		self.batch_generator_test = DatasetBatchGenerator(
			dataset=self.test_dataset,
			frame_size=self.target_size,
			frame_stride=2,
			frame_stride_augment=False,
		)

		num_throws = len(self.dataset.throw_classes)

		self.model = CNN2DRNN(self.target_size, num_throws)
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

	def enable_cnn3drnn(self):
		"""
		Configures a 2D CNN + LSTM model for variable-length sequences in batches.
		Applies 2D CNNs per frame, then processes the sequence with an LSTM.
		"""

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		self.batch_generator_train = DatasetBatchGenerator(
			dataset=self.dataset,
			frame_size=self.target_size,
			frame_stride=2,
			frame_stride_augment=True,
		)

		self.batch_generator_test = DatasetBatchGenerator(
			dataset=self.test_dataset,
			frame_size=self.target_size,
			frame_stride=2,
			frame_stride_augment=False,
		)

		num_throws = len(self.dataset.throw_classes)

		self.model = CNN3DRNN(self.target_size, num_throws, 3)
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
		'cnn2drnn': enable_cnn2drnn,
		'cnn3drnn': enable_cnn3drnn,
	}


def get_memory_usage():
	"""Returns the current memory usage (in MB)."""

	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()

	return mem_info.rss / (1024 * 1024)


def print_gpu_memory_usage(_epoch, _logs):
	"""Prints the GPU memory usage."""

	gpus = tf.config.experimental.list_physical_devices('GPU')
	if not gpus:
		log.warning("No GPUs found.")
		return

	gpu_names = [
		device.name.replace('/physical_device:', '')
		for device in gpus
	]

	for gpu_name in gpu_names:
		mem_info = tf.config.experimental.get_memory_info(gpu_name)
		current_mb = mem_info['current'] / (1024 ** 2)
		peak_mb = mem_info['peak'] / (1024 ** 2)
		print(f"\n-- Memory usage for {gpu_name} --")
		print(f"  Current: {current_mb:.4f} MB")
		print(f"  Peak:    {peak_mb:.4f} MB")

	return
