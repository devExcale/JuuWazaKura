import os
from re import match

from jwk.utils import ts_to_sec


class DatasetInstance:

	def __init__(self, competition: str, dir_dataset: str, dir_segments: str) -> None:
		"""
		Utility class to check the status of a dataset.

		:param competition: Name of the competition.
		:param dir_dataset: Path to the dataset directory.
		:param dir_segments: Path to the segments directory.
		"""

		self.competition: str = competition
		""" Name of the competition. """

		self.dir_dataset: str = dir_dataset
		""" Path to the dataset directory. """

		self.dir_segments: str = dir_segments
		""" Path to the segments directory. """

		self.path_csv = os.path.join(self.dir_dataset, f'{self.competition}.csv')
		""" Path to the CSV file of the dataset. """

		self.path_video = os.path.join(self.dir_segments, f'{self.competition}.mp4')
		""" Path to the video file of the dataset. """

		self.detected_segments: set[int] = set()
		""" The detected segments of the video. """

		self.dir_competition = os.path.join(self.dir_segments, self.competition)
		""" Path to the competition folder in the segments directory. """

		if self.is_dir_competition_present():
			regex_segment_filename = rf'^{self.competition}-(\d+)\.mp4$'

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
							os.listdir(self.dir_competition)
						)
					)
				)
			)

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

		return os.path.exists(self.dir_competition)

	def handler_diff(self) -> set[int]:
		"""
		Returns a set with the segments that are present in the dataset but not in the segments directory.
		The set contents are the milliseconds of the start of the segments.

		:raise FileNotFoundError: If the CSV file is not present.
		:return: Set with the missing segments.
		"""

		if not self.is_csv_present():
			raise FileNotFoundError(f"CSV file not found: {self.path_csv}")

		# Import here to avoid circular imports
		from .ds_handler import DatasetHandler

		# Load dataset with csv file
		handler = DatasetHandler()
		handler.load(self.path_csv)

		# Get all the throws start frames
		ds_segments = set(map(lambda t: int(ts_to_sec(t) * 1000), handler.df['ts_start']))

		# Return the difference between the dataset and the detected segments
		return ds_segments - self.detected_segments
