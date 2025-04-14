import os.path

import click
import numpy as np

from .gipick import main_picker, main_export, compute_gi_histogram, filename_gi_histogram
from .yolov11 import annotate_competition_segments, apply_filter
from ..utils import MyEnv


@click.group(name='devtools')
def cmd_devtools() -> None:
	"""
	Devtools command line interface.
	"""

	return


@cmd_devtools.command(name='gi-hist')
@click.option(
	'--hue-bins', '-h',
	default=180,
	help='Number of bins for hue on gi histogram'
)
@click.option(
	'--sat-bins', '-s',
	default=256,
	help='Number of bins for saturation on gi histogram'
)
def cmd_histogram(hue_bins: int = 180, sat_bins: int = 256) -> None:
	"""
	Generate the Hue-Saturation histogram for all images in the gi folder.

	:param hue_bins: Number of bins for hue on gi histogram
	:param sat_bins: Number of bins for saturation on gi histogram
	"""

	# Set the hue and saturation bins in the environment
	MyEnv.preprocess_hue_bins = hue_bins
	MyEnv.preprocess_sat_bins = sat_bins

	filepath = os.path.join(MyEnv.dataset_source, filename_gi_histogram(hue_bins, sat_bins))
	histogram: np.ndarray | None = None

	# Try loading the histogram file
	if os.path.isfile(filepath):
		try:
			histogram = np.load(filepath)
		except OSError | ValueError:
			# Ignore loading errors, create new histogram
			pass

	# If histogram exists don't create a new one
	if histogram is not None:
		click.echo(f'Histogram(hue={hue_bins},sat={sat_bins}) already exists, skipping histogram generation.')
		return

	# Generate histogram
	histogram = compute_gi_histogram(hue_bins, sat_bins)

	# Save histogram to file
	np.save(filepath, histogram)

	click.echo(f'Histogram(hue={hue_bins},sat={sat_bins}) saved to {filepath}.')

	return


@cmd_devtools.command(name='gi-pick')
def cmd_gi_pick() -> None:
	"""
	Interactively classify gi images.
	"""

	gi_folder = os.path.join(MyEnv.dataset_source, 'gi')
	main_picker(gi_folder)

	return


@cmd_devtools.command(name='gi-export')
def cmd_gi_export() -> None:
	"""
	Export gi images.
	"""

	ds_folder = os.path.join(MyEnv.dataset_clips, 'TLIG2202D1T3')
	gi_folder = os.path.join(MyEnv.dataset_source, 'gi')
	main_export(ds_folder, gi_folder, 'yolo11s.pt')

	return


@cmd_devtools.command(name='yolov11')
def cmd_yolov11() -> None:
	"""
	Annotate competition segments.
	"""

	annotate_competition_segments('yolo11s.pt', 'LIG2202D1T3')

	return


@cmd_devtools.command(name='filter')
def cmd_filter() -> None:
	"""
	Apply filter to competition segments.
	"""

	apply_filter('yolo11s.pt', 'TLIG2202D1T3')

	return


if __name__ == '__main__':
	cmd_devtools()
