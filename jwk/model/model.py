import gc
import logging
import os

import cv2
import numpy as np
import psutil
from keras import Model, Input
from keras.api import backend
from keras.api.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.api.models import Sequential

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

		self.target_frames: int = 16

		self.segments_per_batch: int = 8

		self.model: Model | None = None
		""" Model object. """

		# Load entire dataset (csv records)
		self.dataset.load_all(MyEnv.dataset_source, set(MyEnv.dataset_include), set(MyEnv.dataset_exclude))
		self.dataset.finalize()

		self.batch_generator: DatasetBatchGenerator = DatasetBatchGenerator(
			dataset=self.dataset,
			segments_per_batch=self.segments_per_batch,
			normalize_func=self.normalize_frames,
			morph_funcs=[self.resize_frames, self.resize_flip_frames],
			workers=4,
			use_multiprocessing=True,
		)

		return

	def normalize_frames(self, frames: list[np.ndarray] | np.ndarray) -> np.ndarray:

		# Normalize frames
		frames = [frame / 255. for frame in frames]

		# Normalize frame count
		num_frames = len(frames)
		if num_frames < self.target_frames:

			num_padding = self.target_frames - num_frames
			frames.extend([frames[-1]] * num_padding)

		elif num_frames > self.target_frames:

			indices = np.linspace(0, num_frames - 1, self.target_frames, dtype=int)
			frames = [frames[i] for i in indices]

		# Stack frames into a single array
		data = np.stack(frames, axis=0).astype(np.float32)

		return data

	def resize_frames(self, frames: list[np.ndarray] | np.ndarray) -> np.ndarray:

		resized_frames = np.array([
			cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
			for frame in frames
		])

		return resized_frames

	def resize_flip_frames(self, frames: list[np.ndarray] | np.ndarray) -> np.ndarray:

		flipped_frames = np.array([
			cv2.flip(frame, 1)
			for frame in self.resize_frames(frames)
		])

		return flipped_frames

	def enable_3dcnn(self):

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		input_shape = (self.target_frames, *self.target_size, 3)

		self.model = Sequential([
			Input(shape=input_shape),
			Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
			MaxPooling3D((2, 2, 2)),
			Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
			MaxPooling3D((2, 2, 2)),
			Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
			MaxPooling3D((2, 2, 2)),
			Flatten(),
			Dense(512, activation='relu'),
			Dense(256, activation='relu'),
			Dense(len(self.dataset.throw_classes) + len(self.dataset.tori_classes), activation='softmax'),
		])
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		return

	def fit_model(self, epochs=10):

		self.enable_3dcnn()

		# self.model.fit(
		# 	self.batch_generator,
		# 	epochs=epochs
		# )

		steps = len(self.batch_generator)

		for epoch in range(epochs):

			print(f"Epoch {epoch + 1}/{epochs}")

			for step in range(steps):
				x_batch, y_batch = self.batch_generator[step]
				loss, accuracy = self.model.train_on_batch(x_batch, y_batch)

				memory_usage = get_memory_usage()
				log.info(
					f"  Step {step + 1}/{steps} "
					f"- Loss: {loss:.4f}"
					f", Accuracy: {accuracy:.4f}"
					f", Memory: {memory_usage:.2f} MB"
				)

				# Explicitly call garbage collection
				del x_batch, y_batch
				gc.collect()

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


def get_memory_usage():
	"""Returns the current memory usage (in MB)."""

	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()

	return mem_info.rss / (1024 * 1024)
