import argparse
import json
import logging
import os

from .ds_handler import DatasetHandler
from .ds_downloader import download_dataset

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument(
	'action',
	choices=['stats', 'download'],
	help='Action to perform: stats or download'
)


def main_stats(dir_dataset: str):
	handler = DatasetHandler()
	handler.config()

	# Load all the data
	handler.load_all()

	# Clean the data
	handler.clean_data()

	# Find unknown throws
	unknown_throws = handler.get_unknown_throws()

	# Compute statistics
	stats = handler.compute_stats()

	# Print statistics (beautify)
	print(json.dumps(stats, indent=4))

	# Print unknown throws
	print('Unknown throws:', unknown_throws)


def main_download(dir_dataset: str):
	download_dataset(dir_dataset)


def main():

	logging.basicConfig(level=logging.DEBUG)

	cwd = os.getcwd()
	dir_dataset = os.path.join(cwd, 'dataset')

	args = parser.parse_args()

	if args.action == 'stats':
		main_stats(dir_dataset)
	elif args.action == 'download':
		main_download(dir_dataset)


if __name__ == '__main__':
	main()
