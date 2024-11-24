import json
import os
from typing import Optional, Dict, List, Set

import pandas as pd


class DatasetHandler:

	def __init__(self):
		# Search for dataset folder at project root
		parent_dir = os.path.dirname(os.path.abspath(__file__))
		for _ in range(3):
			parent_dir = os.path.dirname(parent_dir)

		self.directory: Optional[str] = os.path.join(parent_dir, 'dataset')
		"Path to the folder containing the csv files."

		self.df = pd.DataFrame()
		"Main DataFrame to store the data."

		self.columns: Set[str] = set()
		"Columns that must be present in the CSV files."

		self.stats_columns: Set[str] = set()
		"Columns for which statistics must be computed."

		self.throws_known: Set[str] = set()
		"List of known throws."

		self.throws_aliases: Dict[str, str] = {}

	def config(self) -> None:

		# Get json file path
		parent_dir = os.path.dirname(os.path.abspath(__file__))
		config_path = os.path.join(parent_dir, 'dataset_config.json')

		# Read the json file
		with open(config_path, 'r') as file:
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

	def load_all(self) -> None:
		"""
		Loads all the csv files in the directory.
		"""
		for file in os.listdir(self.directory):
			if file.endswith('.csv'):
				self.load(self.directory + '/' + file)
		return

	def load(self, filepath: str) -> None:
		"""
		Loads a csv file.
		:param filepath: Name of the csv file.
		"""

		# Read the CSV file
		loaded_df = pd.read_csv(
			filepath,
			sep=',',
			header=0,
			dtype=str
		)

		# Assert needed columns are present
		missing_cols = self.columns - set(loaded_df.columns)
		if missing_cols:
			filename = filepath.split('/')[-1]
			raise ValueError(
				f"Missing columns in {filename}: {', '.join(missing_cols)} | Present: {', '.join(loaded_df.columns)}")

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
