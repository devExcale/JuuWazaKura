import json
import os.path

import click
import cv2
import numpy as np

from .gipick import main_picker, main_export
from .yolov11 import annotate_competition_segments, apply_filter
from ..model import FrameBox
from ..utils import MyEnv


@click.group(name='devtools')
def cmd_devtools() -> None:
	"""
	Devtools command line interface.
	"""

	return


@cmd_devtools.command(name='gi-hist-gen')
@click.option(
	'--overwrite', '-w',
	default=False,
	is_flag=True,
	help='Overwrite the histogram file if it exists'
)
def cmd_histogram(overwrite: bool) -> None:
	"""
	Generate the Hue-Value histogram for all images in the gi folder.

	:param overwrite: Overwrite the histogram file if it exists
	"""

	# Get histogram if exists
	histogram: np.ndarray | None = FrameBox.get_gi_histogram()

	# If histogram exists don't create a new one
	if histogram is not None and not overwrite:
		click.echo(f'Histogram already exists, skipping histogram generation.')
		return

	# Generate histogram
	histogram = compute_avg_gi_histogram()

	# Save histogram to file
	filepath = os.path.join(MyEnv.dataset_source, FrameBox.FILENAME_GI_HISTOGRAM)
	np.save(filepath, histogram)

	click.echo(f'Histogram saved to {filepath}.')

	return


@cmd_devtools.command(name='gi-hist-test')
def cmd_test_histogram() -> None:
	"""
	Test the histogram generation and loading.
	"""

	# List of subfolders
	gi_folder = os.path.join(MyEnv.dataset_source, 'gi')
	subfolders = ['white', 'blue', 'bin']

	stats = {}

	# Loop through each subfolder
	for subfolder in subfolders:

		# Check if the subfolder exists
		subfolder_path = os.path.join(gi_folder, subfolder)
		if not os.path.exists(subfolder_path):
			continue

		scores = []

		# Loop through each image in the subfolder
		for filename in os.listdir(subfolder_path):
			filepath = os.path.join(subfolder_path, filename)
			if not os.path.isfile(filepath):
				continue

			img = cv2.imread(filepath)
			if img is None:
				continue

			# Compute histogram score
			box = FrameBox(0, 0, img.shape[1], img.shape[0])
			scores.append(box.gi_likelihood(img))

		if not scores:
			continue

		# Compute stats
		substats = {
			'min': np.min(scores),
			'avg': np.mean(scores),
			'max': np.max(scores),
			'median': np.median(scores),
			'std': np.std(scores),
		}
		stats[subfolder] = substats

	if not stats:
		click.echo("No images found in the gi folder.")
		return

	# Print results
	click.echo(json.dumps(stats, indent=4))

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

	apply_filter('SAR2210D2T2FIN')

	return


def compute_avg_gi_histogram() -> np.ndarray:
	"""
	Creates a prototype Hue-Value histogram by averaging histograms from sample judogi images.

	:return: The averaged prototype histogram.
	"""

	# Get subfolders paths
	gi_folders = [
		os.path.join(MyEnv.dataset_source, 'gi', subfolder)
		for subfolder in ('white', 'blue')
	]

	# Get images paths
	images = [
		os.path.join(folder, filename)
		for folder in gi_folders
		for filename in os.listdir(folder)
		if filename.split('.')[-1] in {'jpg', 'png', 'jpeg'}
	]

	list_hists = []

	# Loop over all images
	for img_path in images:

		# Read image
		img = cv2.imread(img_path)
		if img is None:
			continue

		# Compute histogram with FrameBox
		box = FrameBox(0, 0, img.shape[1], img.shape[0])
		hist = box.calc_hv_histogram(img, normalize=False)

		list_hists.append(hist)

	if not list_hists:
		raise FileNotFoundError("No images found in the gi folders to create a histogram.")

	# Compute average histogram (normalized)
	avg_hist = np.mean(list_hists, axis=0)
	cv2.normalize(avg_hist, avg_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

	return avg_hist


if __name__ == '__main__':
	cmd_devtools()
