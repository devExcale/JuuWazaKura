import logging
from math import ceil

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

from .ds_handler import DatasetHandler
from ..utils import get_logger, MyEnv

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())


class DatasetBatchGenerator(Sequence):

	def __init__(
			self,
			dataset: DatasetHandler,
			input_size: tuple[int, int],
			fixed_frames: int | None = None,
			window_frames: int | None = None,
			segments_per_batch: int = 8,
	) -> None:
		"""
		``fixed_frames`` and ``window_frames`` are mutually exclusive.

		:param dataset: Finalized DatasetHandler.
		:param input_size: Size (hxw) of the input frames.
		:param fixed_frames: Number of frames to subsample from the video.
		:param window_frames: Number of frames to use as a sliding window.
		:param segments_per_batch: How many segments to load per batch. The final batch size will be (#segments) x (#transformations).
		"""

		self.segments_per_batch = segments_per_batch
		""" Number of segments to load per batch. """

		self.batch_size = segments_per_batch * 2
		""" Size of each batch. """

		self.input_size = input_size
		""" Size of the input frames (HxW). """

		self.fixed_frames = fixed_frames
		""" Number of frames to subsample from the video. """

		self.window_frames = window_frames
		""" Number of frames to use as a sliding window. """

		if fixed_frames and window_frames:
			raise ValueError("fixed_frames and window_frames are mutually exclusive.")

		if not fixed_frames and not window_frames:
			raise ValueError("Either fixed_frames or window_frames must be set.")

		# Load entire dataset straight away.
		self.ds: DatasetHandler = dataset

		log.debug(
			f"Creating DatasetBatchGenerator with batch size {self.batch_size}, "
			f"expecting {len(self)} batches for {len(self.ds.df) * 2} items"
		)

		return

	def __len__(self) -> int:
		"""
		Number of batches in the dataset.

		:return: Number of batches.
		"""

		return ceil(len(self.ds.df) / self.segments_per_batch)

	def __getitem__(self, batch_idx: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
		"""
		Get the batch at the given index.

		:param batch_idx: Index of the batch.
		:return: Tuple of data and labels arrays for the batch.
		"""

		log.trace(f"Loading batch {batch_idx}")

		start = batch_idx * self.segments_per_batch

		# Cap upper bound at array length; the last batch may be smaller
		# if the total number of items is not a multiple of batch size.
		end = min(start + self.segments_per_batch, len(self.ds.df))

		n_throws = len(self.ds.throw_classes)
		n_tori = len(self.ds.tori_classes)

		list_data = []
		labels_throw = []
		labels_tori = []

		normalization = self.frames_normalize_fixed_time if self.fixed_frames else self.frames_normalize

		for idx in range(start, end):
			x, y = self.ds.xy_train(idx, normalize_x=normalization)

			for morph_func in [self.frames_resize, self.frames_resize_flip]:
				# noinspection PyArgumentList
				list_data.append(morph_func(x))
				labels_throw.append(y[:n_throws])
				labels_tori.append(y[n_throws:])

		batch_data = np.array(list_data)
		batch_throw = np.array(labels_throw)
		batch_tori = np.array(labels_tori)

		if len(batch_data) != self.batch_size or batch_throw.shape[1] != n_throws or batch_tori.shape[1] != n_tori:
			shapes = {
				'x': batch_data.shape,
				'y_throw': batch_throw.shape,
				'y_tori': batch_tori.shape
			}
			raise ValueError(f'Batch {batch_idx} wrongly shaped: {shapes}')

		return batch_data, (batch_throw, batch_tori)

	@staticmethod
	def frames_normalize(frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
		"""
		Normalize the pixel values between ``[0,1]``.

		:param frames: List or array of frames to normalize.
		:return: Normalized frames as a numpy array.
		"""

		# Normalize frames
		max_val = np.max(frames)
		if max_val > 1:
			frames = [frame / 255. for frame in frames]

		# Stack frames into a single array
		data = np.stack(frames, axis=0).astype(np.float32)

		return data

	def frames_normalize_fixed_time(self, frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
		"""
		Normalize the pixel values between ``[0,1]`` and ensure the number of frames is fixed by subsampling or padding.

		:param frames:
		:return:
		"""

		# Normalize frames
		max_val = np.max(frames)
		if max_val > 1:
			frames = [frame / 255. for frame in frames]

		# Normalize frame count
		num_frames = len(frames)
		if num_frames < self.fixed_frames:

			num_padding = self.fixed_frames - num_frames
			frames.extend([frames[-1]] * num_padding)

		elif num_frames > self.fixed_frames:

			indices = np.linspace(0, num_frames - 1, self.fixed_frames, dtype=int)
			frames = [frames[i] for i in indices]

		# Stack frames into a single array
		data = np.stack(frames, axis=0).astype(np.float32)

		return data

	def frames_resize(self, frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
		"""
		Resize the frames to the specified input size.

		:param frames: List or array of frames to resize.
		:return: Resized frames as a numpy array.
		"""

		if frames[0].shape[:2] != self.input_size:
			resized_frames = np.array([
				cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
				for frame in frames
			])
		else:
			resized_frames = np.array(frames)

		return resized_frames

	def frames_resize_flip(self, frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
		"""
		Resize and flip the frames.

		:param frames: List or array of frames to resize and flip.
		:return: Resized and flipped frames as a numpy array.
		"""

		flipped_frames = np.array([
			cv2.flip(frame, 1)
			for frame in self.frames_resize(frames)
		])

		return flipped_frames

	@staticmethod
	def frames_flip(frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
		"""
		Flip the frames.

		:param frames: List or array of frames to flip.
		:return: Flipped frames as a numpy array.
		"""

		flipped_frames = np.array([
			cv2.flip(frame, 1)
			for frame in frames
		])

		return flipped_frames
