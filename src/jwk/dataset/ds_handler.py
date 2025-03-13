import json
import logging
import os
from re import match
from typing import Generator, Hashable

import cv2
import numpy as np
import pandas as pd
from keras.api.utils import to_categorical

from ..utils import MyEnv, get_logger, regex_ts, ts_to_sec

log: logging.Logger = get_logger(__name__, MyEnv.log_level())


class DatasetHandler:

	def __init__(self) -> None:

		self.df = pd.DataFrame()
		""" Main DataFrame to store the data. """

		self.columns: set[str] = set()
		""" Columns that must be present in the CSV files. """

		self.stats_columns: set[str] = set()
		""" Columns for which statistics must be computed. """

		self.known_tori: set[str] = set()
		""" Set of possible tori values. """

		self.known_throws: set[str] = set()
		""" Set of possible throw name values. """

		self.throw_from_alias: dict[str, str] = {}
		""" Mapping of aliases to throw name. Label-normalized. """

		self.throw_to_group: dict[str, str] = {}
		""" Mapping of throw name to a possible throw group. Label-normalized. """

		self.throw_classes: list[str] = []
		""" Set of all actually loaded throw labels, sorted alphabetically. """

		self.tori_classes: list[str] = []
		""" Set of all actually loaded tori labels, sorted alphabetically. """

		self.throw_onehot: dict[str, np.ndarray] = {}
		""" Mapping from throw name to the respective onehot encoding. """

		self.tori_onehot: dict[str, np.ndarray] = {}
		""" Mapping from tori value to the respective onehot encoding. """

		self.finalized: bool = False

		self.__apply_config__()

		return

	def __apply_config__(self, config_json_path: str | None = None) -> None:

		# Get json file path
		if config_json_path is None:
			parent_dir = os.path.dirname(os.path.abspath(__file__))
			config_json_path = os.path.join(parent_dir, 'dataset_config.json')

		# Read the json file
		with open(config_json_path, 'r') as file:
			config = json.load(file)

		# Load dataset configuration
		self.columns = set(config['columns'])
		self.stats_columns = set(config['stats_columns'])
		self.known_tori = set(normalize_label(s) for s in config['known_tori'])
		self.known_throws = set(normalize_label(s) for s in config['known_throws'])

		# Create mapping from throw alias to throw name (e.g., 'sawari seoi' -> 'sawari seoi nage')
		self.throw_from_alias = {
			normalize_label(k): normalize_label(v)
			for k, v in config['throw_aliases'].items()
		}

		# Create mapping from throw name to respective group (e.g., 'sawari seoi nage' -> 'seoi nage')
		for throw_group, throw_list in config['throw_groups'].items():
			for throw in throw_list:
				self.throw_to_group[normalize_label(throw)] = normalize_label(throw_group)

		# Set dataframe
		self.df = pd.DataFrame(columns=list(self.columns))

		return

	def __finalize_barrier__(self) -> None:
		"""
		Checks whether the data is finalized, and raises an error if it is.
		"""

		if self.finalized:
			raise ValueError("Data is already finalized")

		return

	def load_all(self, directory: str) -> None:
		"""
		Loads all the csv files in the dataset directory.

		See also: load()
		"""

		self.__finalize_barrier__()

		# Initialize loaded dataframes list
		concat_list: list[pd.DataFrame] = [self.df]

		# Load all dataframes in directory
		for file in os.listdir(directory):
			if file.endswith('.csv'):
				self.load(os.path.join(directory, file), concat_list=concat_list)

		# Concatenate all dataframes
		self.df = pd.concat(concat_list, ignore_index=True)

		return

	def load(self, csv_path: str, concat_list: list[pd.DataFrame] | None = None) -> None:
		"""
		Loads a csv file and appends it to the internal dataframe, or adds it to the concat list if present.
		If the concat list is present, the data is not automatically appended to the main DataFrame.

		:param csv_path: Path to the csv file.
		:param concat_list: List
		"""

		self.__finalize_barrier__()

		# Read the CSV file
		loaded_df = pd.read_csv(
			csv_path,
			sep=',',
			header=0,
			dtype=str
		)

		# Assert needed columns are present
		missing_cols = self.columns - set(loaded_df.columns)
		if missing_cols:
			filename = csv_path.split('/')[-1]
			str_missing_cols = ', '.join(missing_cols)
			str_present_cols = ', '.join(loaded_df.columns)
			raise ValueError(f"Missing columns in {filename}: [{str_missing_cols}] | Present: [{str_present_cols}]")

		# Append the data to the main DataFrame
		if concat_list is None:
			self.df = pd.concat([self.df, loaded_df], ignore_index=True)
		else:
			concat_list.append(loaded_df)

		return

	def finalize(self) -> None:
		"""
		Checks whether the loaded data is valid, morphs it if necessary, and locks it from further modifications.

		:raise ValueError: If the data is not valid.
		:return: ``None``
		"""

		# Check if already finalized
		if self.finalized:
			return

		# Loop over rows
		for index, row in self.df.iterrows():
			self.__finalize_row__(index, row)

		# Update actual labels
		self.throw_classes = list(sorted(
			self.df['throw']
			.map(normalize_label)
			.map(lambda t: self.throw_from_alias.get(t, t))
			.map(lambda t: self.throw_to_group.get(t, t))
			.unique()
		))
		self.tori_classes = list(sorted(
			self.df['tori']
			.map(normalize_label)
			.unique()
		))

		len_throws = len(self.throw_classes)
		len_tori = len(self.tori_classes)

		# Build onehot encodings (throw)
		for i, throw in enumerate(self.throw_classes):
			self.throw_onehot[throw] = to_categorical(i, len_throws)

		# Build onehot encodings (tori)
		for i, tori in enumerate(self.tori_classes):
			self.tori_onehot[tori] = to_categorical(i, len_tori)

		# Prevent further modifications
		self.finalized = True

		return

	def __finalize_row__(self, index: Hashable, row: pd.Series) -> None:
		"""
		Performs final checks on a row of the dataset and morphs it if necessary.

		:param index: Index of the row.
		:param row: Row data.
		:return: ``None``
		"""

		# Get fields
		name = row['competition']
		ts_start = row['ts_start']
		ts_end = row['ts_end']
		throw = normalize_label(row['throw'])
		tori = normalize_label(row['tori'])

		# Verbose row identifier
		row_id = f"Competition={name} | Start={ts_start} | End={ts_end}"

		# Check invalid timestamps formats
		if not match(regex_ts, ts_start) or not match(regex_ts, ts_end):
			raise ValueError(f"Invalid timestamps: {row_id}")

		# Check invalid timestamps order
		start = ts_to_sec(ts_start)
		end = ts_to_sec(ts_end)
		if start >= end:
			raise ValueError(f"Invalid timestamps order: {row_id}")

		# Check invalid tori names
		if tori not in self.known_tori:
			raise ValueError(f"Unknown tori label: {row_id}")

		# Map throw alias
		throw = self.throw_from_alias.get(throw, throw)

		# Check invalid throw names
		if throw not in self.known_throws:
			raise ValueError(f"Unknown throw label: {row_id}")

		# Map throw to group
		throw = self.throw_to_group.get(throw, throw)

		# Update the row
		self.df.at[index, 'throw'] = throw

		return

	def get_unknown_throws(self) -> set[str]:
		"""
		Returns a list of throws that are not in the known throws.
		:return: List of unknown throws.
		"""

		throws = {
			throw
			for throw in self.df['throw'].unique()
			if throw not in self.known_throws
		}

		# Remove mapped groups from unknown throws
		return throws - set(self.throw_to_group.values())

	def compute_stats(self) -> dict[str, dict[str, int]]:
		"""
		Computes statistics from the data.
		:return: Dictionary containing the computed statistics.
		"""

		stats = {
			column: self.df[column].value_counts().to_dict()
			for column in self.stats_columns
		}

		return stats

	def xy_generator(self, raise_on_error: bool = False) -> Generator[tuple[list[np.ndarray], str, str], None, None]:
		"""
		Generator that loads all video data and yields it with the corresponding labels.
		The frames come as-is from cv2, with no preprocessing.

		:param raise_on_error: Whether to raise an error if a video file can't be loaded.
		:return: Generator for the dataset data ``(frames,throw,tori)``.
		"""

		comp_skips: list[str] = []

		# Check for unsegmented competitions
		for competition in self.df['competition'].unique():
			if not os.path.exists(os.path.join(MyEnv.dataset_clips, competition)):
				log.warning(f"Competition not segmented: {competition}")
				comp_skips.append(competition)

		# Loop over the dataset
		for index, row in self.df.iterrows():

			# Record attributes
			competition = row['competition']
			throw = row['throw']
			tori = row['tori']
			ts_start = row['ts_start']

			# Skip unsegmented competitions
			if competition in comp_skips:
				continue

			# Get video path
			ms_start = int(ts_to_sec(ts_start) * 1000)
			clip_name = f'{competition}-{ms_start}.mp4'
			video_path = os.path.join(MyEnv.dataset_clips, competition, clip_name)

			frames = []

			# Load video data
			cap = cv2.VideoCapture(video_path)
			while cap.isOpened():

				# Read frame
				ret, frame = cap.read()

				# Break if no frame
				if not ret:
					break

				frames.append(frame)

			cap.release()

			# Handle loading error
			if not frames:
				msg = f"Error loading video: {video_path}"
				if raise_on_error:
					raise ValueError(msg)
				else:
					log.warning(msg)
					continue

			# Yield the data
			yield frames, throw, tori

		# End generator
		return


def normalize_label(name: str) -> str:
	"""
	Normalizes a label value: lowercase, no leading/trailing whitespaces, and spaces replaced by underscores.

	:param name: Name to normalize.
	:return: Standardized name.
	"""

	# Strip leading and trailing whitespaces
	name = name.strip()

	# Lowercase throw name
	name = name.lower()

	# Replace spaces with underscores
	name = name.replace(' ', '_')

	return name
