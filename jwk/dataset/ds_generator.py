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
			frame_size: tuple[int, int],
			frame_stride: int = 1,
	) -> None:
		"""

		:param dataset: Finalized DatasetHandler.
		:param frame_size: Size (hxw) of the input frames.
		:param frame_stride: Stride for subsampling frames.
		"""

		self.frame_size = frame_size
		""" Size of the input frames (H, W). """

		self.frame_stride = frame_stride
		""" Stride for subsampling frames. """

		self.batch_size = 1
		""" Batch size for the generator. """

		self.input_shape = (1, None, *self.frame_size, 3)
		""" Input shape of the model (B, T, H, W, C). """

		# Load entire dataset straight away.
		self.ds: DatasetHandler = dataset

		self.transforms: list[callable] = [
			self.transform_resize,
			self.transform_resize_flip,
		]
		""" List of transformations to apply to the frames. """

		log.debug(
			f"Creating DatasetBatchGenerator with batch size {1}, "
			f"expecting {len(self)} batches for {len(self.ds.df) * len(self.transforms)} items"
		)

		return

	def __len__(self) -> int:
		"""
		Number of batches in the dataset.

		:return: Number of batches.
		"""

		return len(self.ds.df) * len(self.transforms)

	def __getitem__(self, batch_idx: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
		"""
		Get the batch at the given index.

		:param batch_idx: Index of the batch.
		:return: Tuple of data and labels arrays for the batch.
		"""

		log.trace(f"Loading batch {batch_idx}")

		n_throws = len(self.ds.throw_classes)
		n_tori = len(self.ds.tori_classes)
		n_transforms = len(self.transforms)

		idx_segment = batch_idx // n_transforms
		idx_transform = batch_idx % n_transforms

		list_data = []
		labels_throw = []
		labels_tori = []

		# Get segment
		x, y = self.ds.xy_train(idx_segment, normalize_x=self.transform_normalize)

		list_data.append(self.transforms[idx_transform](x))
		labels_throw.append(y[:n_throws])
		labels_tori.append(y[n_throws:])

		# Ensure videos have loaded
		if len(list_data) == 0:
			raise ValueError(f'Batch {batch_idx} is empty.')

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

		return batch_data.copy(), {'throw': batch_throw.copy(), 'tori': batch_tori.copy()}

	def transform_normalize(self, frames: np.ndarray) -> np.ndarray:
		"""
		Normalize the pixel values between ``[0,1]`` and halve the framerate if specified.

		:param frames: List or array of frames to normalize.
		:return: Normalized frames as a numpy array.
		"""

		# Normalize frames
		max_val = np.max(frames)
		frames = frames.astype(np.float32)
		if max_val > 1:
			frames /= 255.

		# Subsample frames by the specified stride
		if self.frame_stride != 1:
			frames = frames[::self.frame_stride]

		return frames

	def transform_resize(self, frames: np.ndarray) -> np.ndarray:
		"""
		Resize the frames to the specified input size.

		:param frames: Array of frames to resize.
		:return: Resized frames as a numpy array.
		"""

		if frames[0].shape[:2] != self.frame_size:
			resized_frames = np.array([
				cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
				for frame in frames
			])
		else:
			resized_frames = frames

		return resized_frames

	def transform_resize_flip(self, frames: np.ndarray) -> np.ndarray:
		"""
		Resize and flip the frames.

		:param frames: List or array of frames to resize and flip.
		:return: Resized and flipped frames as a numpy array.
		"""

		flipped_frames = np.array([
			cv2.flip(frame, 1)
			for frame in self.transform_resize(frames)
		])

		return flipped_frames

	@staticmethod
	def frames_flip(frames: np.ndarray) -> np.ndarray:
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
