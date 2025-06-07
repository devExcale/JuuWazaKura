import os
from re import match

from ..utils import ts_to_sec

from .ds_orchestrator import DatasetOrchestrator


class LiveFootageInstance:

	def __init__(self, code_footage: str, dir_dataset: str, dir_segments: str) -> None:
		"""
		Utility class to check the status of live-footage.

		:param code_footage: Live-footage's code.
		:param dir_dataset: Path to the dataset (csv) directory.
		:param dir_segments: Path to the segments (mp4) directory.
		"""

		self.code_footage: str = code_footage
		""" Code of the live-footage. """

		self.dir_dataset: str = dir_dataset
		""" Path to the dataset directory containing the csv files. """

		self.dir_segments: str = dir_segments
		""" Path to the dataset directory containing the segment folders. """

		self.dir_segments_footage = os.path.join(self.dir_segments, self.code_footage)
		""" Path to the directory containing the footage's segments. """

		self.path_csv = os.path.join(self.dir_dataset, f'{self.code_footage}.csv')
		""" Path to the CSV file of the dataset. """

		self.path_video = os.path.join(self.dir_segments, f'{self.code_footage}.mp4')
		""" Path to the video file of the footage. """

		self.detected_segments: set[int] = set()
		""" Set of the segments detected in the footage. """

		if self.is_dir_competition_present():
			regex_segment_filename = rf'^{self.code_footage}-(\d+)\.mp4$'

			self.detected_segments = set(
				map(
					# Extract the milliseconds from the filename
					lambda m: int(m.group(1)),
					filter(
						# Look for matches
						lambda m: m is not None,
						map(
							# Map filenames to regex match objects
							lambda filename: match(regex_segment_filename, filename),
							os.listdir(self.dir_segments)
						)
					)
				)
			)

		self.ds_orch: DatasetOrchestrator | None = None
		""" Orchestrator with the footage's data. """

		return

	def is_csv_present(self) -> bool:
		"""
		Checks whether the CSV file is present.

		:return: Whether the CSV file is present.
		"""

		return os.path.exists(self.path_csv)

	def is_video_present(self) -> bool:
		"""
		Checks whether the video file is present.

		:return: Whether the video file is present.
		"""

		return os.path.exists(self.path_video)

	def is_dir_competition_present(self) -> bool:
		"""
		Checks whether the competition folder is present.

		:return: Whether the competition folder is present.
		"""

		return os.path.exists(self.dir_segments)

	def load_dataset(self) -> None:
		"""
		Load the footage's data from the CSV file.

		:raise FileNotFoundError: If the CSV file is not found.
		:raise ValueError: If the dataset data is not valid.
		:return: ``None``
		"""

		if not self.is_csv_present():
			raise FileNotFoundError(f"CSV file not found: {self.path_csv}")

		if self.ds_orch is None:
			self.ds_orch = DatasetOrchestrator()
			self.ds_orch.load(self.path_csv)

		self.ds_orch.finalize()

		return

	def missing_segments(self) -> set[int]:
		"""
		Returns a set with the segments that are found in the dataset but not in the footage's segments directory.
		The set contents are the milliseconds the segments' start timestamps.

		:raise FileNotFoundError: If the CSV file is not found.
		:raise ValueError: If the dataset data is not valid.
		:return: Set with the missing segments.
		"""

		if not self.is_csv_present():
			raise FileNotFoundError(f"CSV file not found: {self.path_csv}")

		# Ensure dataset is loaded and valid
		self.load_dataset()

		# Get all the throws start frames
		ds_segments = set(map(lambda t: int(ts_to_sec(t) * 1000), self.ds_orch.df['ts_start']))

		# Return the difference between the dataset and the detected segments
		return ds_segments - self.detected_segments
