import json
import logging
import os
from typing import Optional, Dict, Set

import pandas as pd

from ..utils import MyEnv, get_logger

log: logging.Logger = get_logger(__name__, MyEnv.log_level())


class DatasetHandler:

	def __init__(self) -> None:

		self.df = pd.DataFrame()
		""" Main DataFrame to store the data. """

		self.columns: Set[str] = set()
		""" Columns that must be present in the CSV files. """

		self.stats_columns: Set[str] = set()
		""" Columns for which statistics must be computed. """

		self.throws_known: Set[str] = set()
		""" List of known throws. """

		self.throws_aliases: Dict[str, str] = {}
		""" Mapping of throw aliases. """

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

		# Set the configuration variables
		self.columns = set(config['columns'])
		self.stats_columns = set(config['stats_columns'])
		self.throws_known = set(std_throw_name(s) for s in config['throws_known'])
		self.throws_aliases = {
			std_throw_name(k): std_throw_name(v)
			for k, v in config['throws_aliases'].items()
		}

		# Set dataframe
		self.df = pd.DataFrame(columns=list(self.columns))

		return

	def load_all(self, directory: str) -> None:
		"""
		Loads all the csv files in the dataset directory.

		See also: load()
		"""

		for file in os.listdir(directory):
			if file.endswith('.csv'):
				self.load(os.path.join(directory, file))

		return

	def load(self, csv_path: str) -> None:
		"""
		Loads a csv file and appends it to the internal dataframe.

		:param csv_path: Path to the csv file.
		"""

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
		# TODO: replace concat with better approach
		self.df = pd.concat([self.df, loaded_df], ignore_index=True)

		return

	def clean_data(self):
		"""
		Cleans the data in the DataFrame.
		"""

		# Loop over all rows
		for index, row in self.df.iterrows():

			# Get throw name
			name = row['throw']

			# Standardize the throw names
			name = std_throw_name(name)

			# Check for aliases
			alias = self.throws_aliases.get(name)
			if alias:
				name = alias

			# Reassign name
			self.df.at[index, 'throw'] = name

	def get_unknown_throws(self) -> Set[str]:
		"""
		Returns a list of throws that are not in the known throws.
		:return: List of unknown throws.
		"""

		throws = {
			throw
			for throw in self.df['throw'].unique()
			if throw not in self.throws_known
		}

		return throws

	def compute_stats(self) -> Dict[str, Dict[str, int]]:
		"""
		Computes statistics from the data.
		:return: Dictionary containing the computed statistics.
		"""

		stats = {
			column: self.df[column].value_counts().to_dict()
			for column in self.stats_columns
		}

		return stats

	def should_download_video(self, name: str) -> bool:
		"""
		Check whether a video should be downloaded.
		A video should be downloaded if there exists a corresponding csv file, the video does not exist
		and all video segments don't exist.

		:param name: Filename of the video (no extension)
		:return: Whether the video should be downloaded
		"""

		# Check whether a corresponding csv exists
		path_csv = os.path.join(self.dir_dataset, f'{name}.csv')
		if not os.path.exists(path_csv):
			return False

		# Check whether the video already exists
		path_video = os.path.join(self.dir_segments, f'{name}.mp4')
		if os.path.exists(path_video):
			return False

		# Check whether the segments directory exists
		if not os.path.exists(os.path.join(self.dir_segments, name)):
			return True

		# Load dataset with csv file
		ds_handler = DatasetHandler()
		ds_handler.load(path_csv)

		# Get all the throws start frames
		ds_start_ms = set(map(lambda t: int(ts_to_sec(t) * 1000), ds_handler.df['ts_start']))

		regex_segment_filename = rf'^{name}-(\d)+\.mp4$'

		# Look for video segments on filepath
		os_start_ms = set(
			map(
				# Extract the milliseconds from the filename
				lambda m: int(m.group(1)),
				filter(
					# Look for matches
					lambda m: m is not None,
					map(
						# Map filenames to regex match objects
						lambda filename: match(regex_segment_filename, filename),
						os.listdir(os.path.join(self.dir_segments, name))
					)
				)
			)
		)

		# If there are missing segments, download the video
		return bool(ds_start_ms - os_start_ms)


def std_throw_name(name: str) -> str:
	"""
	Standardizes the name of a throw.
	:param name: Name of the throw.
	:return: Standardized name.
	"""

	# Strip leading and trailing whitespaces
	name = name.strip()

	# Lowercase throw name
	name = name.lower()

	# Replace spaces with underscores
	name = name.replace(' ', '_')

	return name
