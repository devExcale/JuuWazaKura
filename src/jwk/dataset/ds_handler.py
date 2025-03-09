import json
import logging
import os
from re import match
from typing import Optional, Dict, Set

import pandas as pd

from ..utils import MyEnv, get_logger, regex_ts, ts_to_sec

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

	def validate_data(self) -> None:
		"""
		Checks whether all the loaded data is valid.

		:raise ValueError: If the data is not valid.
		:return: `None`
		"""

		# Loop over rows
		for index, row in self.df.iterrows():

			# Get fields
			name = row['competition']
			ts_start = row['ts_start']
			ts_end = row['ts_end']

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

			# TODO: integrate tori and throw

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
