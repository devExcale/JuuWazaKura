import json

import click

from .ds_downloader import DatasetDownloader
from .ds_orchestrator import DatasetOrchestrator
from ..utils import MyEnv


@click.group(name="ds")
def cmd_dataset():
	"""
	Dataset command line interface.
	"""

	return


@cmd_dataset.command(name="stats")
@click.option(
	'--grouped', '-g',
	default=False,
	is_flag=True,
	help='Whether to group similar throws together.'
)
def cmd_stats(grouped: bool = False) -> None:
	"""
	Compute and display dataset statistics.

	:param grouped: Whether to group similar throws together.
	:return: ``None``
	"""

	# Initialize dataset
	handler = DatasetOrchestrator()

	# Load all the data
	handler.load_all(MyEnv.dataset_source)
	handler.finalize(group_throws=grouped)

	# Compute statistics
	stats = handler.compute_stats()

	# Add unknown throws
	stats['unknown'] = list(handler.get_unknown_throws())

	# Print statistics (beautify)
	click.echo(json.dumps(stats, indent=4))

	return


@cmd_dataset.command(name="download")
def cmd_download():
	"""
	Download dataset segments asynchronously.
	"""

	with DatasetDownloader(MyEnv.dataset_source, MyEnv.dataset_livefootage, MyEnv.dataset_segments) as downloader:
		downloader.download_segment_all_async()

	return


@cmd_dataset.command(name="ytformats")
def cmd_ytformats():
	"""
	Display YouTube video formats for dataset videos.
	"""
	formats = {}

	with DatasetDownloader(MyEnv.dataset_source, MyEnv.dataset_livefootage, MyEnv.dataset_segments) as downloader:
		for yt_id in downloader.yt_video_ids.values():
			formats[yt_id] = downloader.yt_find_format(yt_id)

	click.echo(json.dumps(formats, indent=4))

	return


if __name__ == "__main__":
	cmd_dataset()
