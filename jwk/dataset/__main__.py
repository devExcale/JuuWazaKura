import json

import click

from .ds_downloader import DatasetDownloader
from .ds_handler import DatasetHandler
from ..utils import MyEnv


@click.group(name="ds")
def cmd_dataset():
	"""
	Dataset command line interface.
	"""

	return


@cmd_dataset.command(name="stats")
def cmd_stats():
	"""
	Compute and display dataset statistics.
	"""

	# Initialize dataset
	handler = DatasetHandler()

	# Load all the data
	handler.load_all(MyEnv.dataset_source)
	handler.finalize()

	# Find unknown throws
	unknown_throws = handler.get_unknown_throws()

	# Compute statistics
	stats = handler.compute_stats()

	# Print statistics (beautify)
	click.echo(json.dumps(stats, indent=4))

	# Print unknown throws
	click.echo(f"Unknown throws: {unknown_throws}")

	return


@cmd_dataset.command(name="download")
def cmd_download():
	"""
	Download dataset segments asynchronously.
	"""

	with DatasetDownloader(MyEnv.dataset_source, MyEnv.dataset_clips) as downloader:
		downloader.download_segment_all_async()

	return


@cmd_dataset.command(name="ytformats")
def cmd_ytformats():
	"""
	Display YouTube video formats for dataset videos.
	"""
	formats = {}

	with DatasetDownloader(MyEnv.dataset_source, MyEnv.dataset_clips) as downloader:
		for yt_id in downloader.yt_video_ids.values():
			formats[yt_id] = downloader.yt_find_format(yt_id)

	click.echo(json.dumps(formats, indent=4))

	return


if __name__ == "__main__":
	cmd_dataset()
