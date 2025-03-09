import argparse
import json

from jwk.utils import MyEnv
from .ds_downloader import DatasetDownloader
from .ds_handler import DatasetHandler

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument(
	'action',
	choices=['stats', 'download', 'ytformats'],
	help='Action to perform',
)


def main_stats():
	# Initialize dataset
	handler = DatasetHandler()

	# Load all the data
	handler.load_all(MyEnv.dataset_source)

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
	with DatasetDownloader(MyEnv.dataset_source, MyEnv.dataset_clips) as downloader:
		downloader.download_segment_all_async()

	return


def main_ytformats():
	formats: dict[str, str] = {}

	with DatasetDownloader(MyEnv.dataset_source, MyEnv.dataset_clips) as downloader:
		for yt_id in downloader.yt_video_ids.values():
			formats[yt_id] = downloader.yt_find_format(yt_id)

	print(json.dumps(formats, indent=4))

	return


def main():
	switch = {
		'stats': main_stats,
		'download': main_download,
		'ytformats': main_ytformats,
	}

	args = parser.parse_args()
	main_method = switch.get(args.action, parser.print_help)

	main_method()

	return


if __name__ == '__main__':
	main()
