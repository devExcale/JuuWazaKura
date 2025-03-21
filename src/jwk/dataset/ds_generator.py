import logging
from collections.abc import Callable
from math import ceil

import numpy as np
from keras.api.utils import PyDataset

from .ds_handler import DatasetHandler
from ..utils import get_logger, MyEnv

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())


class DatasetBatchGenerator(PyDataset):

	def __init__(
			self,
			dataset: DatasetHandler,
			normalize_func: Callable[[list[np.ndarray]], np.ndarray],
			morph_funcs: list[Callable[[np.ndarray], np.ndarray]],
			workers: int = 1,
			use_multiprocessing: bool = False,
			max_queue_size: int = 10,
			segments_per_batch: int = 8,
	) -> None:
		"""
		:param dataset: Finalized DatasetHandler.
		:param normalize_func: Function to apply to the list of frames to return the normalized frames as an ndarray.
		:param morph_funcs: List of functions to apply singularly to the segments that return the transformed segment.
		:param workers: Number of workers to use in multithreading or multiprocessing.
		:param use_multiprocessing: Whether to use Python multiprocessing for parallelism.
		:param max_queue_size: Maximum number of batches to keep in the queue when iterating over the dataset in a multithreaded or multiprocessed setting.
		:param segments_per_batch: How many segments to load per batch. The final batch size will be (#segments) x (#transformations).
		"""

		super().__init__(workers, use_multiprocessing, max_queue_size)

		self.segments_per_batch = segments_per_batch
		""" Number of segments to load per batch. """

		if not morph_funcs:
			raise ValueError("At least one morph function must be provided.")

		self.batch_size = segments_per_batch * len(morph_funcs)
		""" Size of each batch. """

		self.normalize_func = normalize_func
		""" Function to normalize the frames. """

		self.morph_funcs = morph_funcs
		""" List of functions to apply to the frames. """

		# Load entire dataset straight away.
		self.ds: DatasetHandler = dataset

		log.debug(
			f"Creating DatasetBatchGenerator with batch size {self.batch_size}, "
			f"expecting {len(self)} batches for {len(self.ds.df) * len(morph_funcs)} items"
		)

		return

	def __len__(self) -> int:
		# Return number of batches.
		return ceil(len(self.ds.df) / self.segments_per_batch)

	def __getitem__(self, batch_idx: int) -> tuple[np.ndarray, np.ndarray]:
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

		x_list = []
		y_list = []

		for idx in range(start, end):
			x, y = self.ds.xy_train(idx, normalize_x=self.normalize_func)

			for morph_func in self.morph_funcs:
				x_list.append(morph_func(x))
				y_list.append(y)

		x_batch = np.array(x_list)
		y_batch = np.array(y_list)

		return x_batch, y_batch
