import argparse
import json
import logging
import os

from .ds_downloader import DatasetDownloader
from .ds_handler import DatasetHandler

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument(
	'action',
	choices=['stats', 'download', 'ytformats'],
	help='Action to perform',
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
	with DatasetDownloader(dir_dataset) as downloader:
		downloader.main_dwnl_clip_async()


def main_ytformats(dir_dataset: str):
	formats: dict[str, str] = {}

	with DatasetDownloader(dir_dataset) as downloader:
		for yt_id in downloader.yt_video_ids.values():
			formats[yt_id] = downloader.yt_find_format(yt_id)

	print(json.dumps(formats, indent=4))


def main():
	logging.basicConfig(level=logging.DEBUG)

	cwd = os.getcwd()
	dir_dataset = os.path.join(cwd, 'dataset')

	args = parser.parse_args()

	if args.action == 'stats':
		main_stats(dir_dataset)
	elif args.action == 'download':
		main_download(dir_dataset)
	elif args.action == 'ytformats':
		main_ytformats(dir_dataset)


if __name__ == '__main__':
	main()
