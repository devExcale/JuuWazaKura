from typing import Generator

import cv2
import numpy as np
from keras import Model
from keras.api.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.api.models import Sequential

from ..dataset import DatasetHandler
from ..utils import MyEnv


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

		self.model: Model | None = None
		""" Model object. """

		# Load entire dataset (csv records)
		self.dataset.load_all(MyEnv.dataset_source)
		self.dataset.finalize()

		return

	def xy_generator(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:

		# Load and normalize video data
		for frames, throw, tori in self.dataset.xy_generator():

			# Normalize frames
			frames = [cv2.resize(frame, self.target_size) / 255. for frame in frames]

			# Normalize frame count
			num_frames = len(frames)
			if num_frames < self.target_frames:

				num_padding = self.target_frames - num_frames
				frames.extend([frames[-1]] * num_padding)

			elif num_frames > self.target_frames:

				indices = np.linspace(0, num_frames - 1, self.target_frames, dtype=int)
				frames = [frames[i] for i in indices]

			# Stack frames into a single array
			data = np.stack(frames, axis=0)

			# Generate data label
			label = np.concatenate((
				self.dataset.throw_onehot[throw],
				self.dataset.tori_onehot[tori],
			))

			# Yield to the training data
			yield data, label

		return

	def enable_3dcnn(self):

		if self.model is not None:
			raise AssertionError("There's another model already enabled.")

		input_shape = (self.target_frames, *self.target_size, 3)

		self.model = Sequential([
			Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
			MaxPooling3D((2, 2, 2)),
			Conv3D(64, (3, 3, 3), activation='relu'),
			MaxPooling3D((2, 2, 2)),
			Conv3D(128, (3, 3, 3), activation='relu'),
			MaxPooling3D((2, 2, 2)),
			Flatten(),
			Dense(512, activation='relu'),
			Dense(256, activation='relu'),
			Dense(len(self.dataset.throw_classes) + len(self.dataset.tori_classes), activation='softmax')
		])
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		return

	def fit_model(self, epochs=10, batch_size=4):
		self.enable_3dcnn()
		self.model.fit(self.xy_generator(), epochs=epochs, batch_size=batch_size)

		return

	def save_weights(self, filepath):
		self.model.save_weights(filepath)

		return
