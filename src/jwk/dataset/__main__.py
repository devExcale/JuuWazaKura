import argparse
import json
import logging
import os

from .ds_handler import DatasetHandler
from .ds_downloader import DatasetDownloader

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument(
	'action',
	choices=['stats', 'download'],
	help='Action to perform: stats or download'
)


def main_stats():
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


def main_download():
	with DatasetDownloader() as downloader:
		downloader.main_dwnl_clip_async()


def main():

	logging.basicConfig(level=logging.DEBUG)

	args = parser.parse_args()

	switch = {
		'stats': main_stats,
		'download': main_download
	}

	main_method = switch.get(args.action, parser.print_help)
	main_method()


if __name__ == '__main__':
	main()
