import os

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..dataset import DatasetOrchestrator
from ..model import FrameBox
from ..utils import MyEnv, ts_to_sec


@click.group(name='plot')
def cmd_plot() -> None:
	"""
	Plotting command line interface.
	This module provides commands to plot Hue/Value histograms from images.
	"""

	return


stat_keys = {
	'throw': 'Numerosity',
	'duration': 'Total Duration (seconds)',
	'average': 'Average Duration (seconds)',
}


@cmd_plot.command(name='stats')
@click.option(
	'--output', '-o',
	default=None,
	help='Path to save the output plot image. If not set, the plot will be displayed on the screen.',
)
@click.option(
	'--stat', '-t',
	default='throw',
	type=click.Choice(list(stat_keys.keys()), case_sensitive=False),
	help='Statistic to plot: "throw" for numerosity, "duration" for duration (default: throw).',
)
@click.option(
	'--sort', '-s',
	default=0,
	type=click.IntRange(0, 1),
	help='Sort parameter: 0 - Name, 1 - Numerosity (default: 0).',
)
@click.option(
	'--grouped', '-g',
	default=False,
	is_flag=True,
	help='Whether to apply aliases and group similar throws together.',
)
def cmd_plot_stats(
		output: str | None = None,
		stat: str = 'throw',
		sort: int = 0,
		grouped: bool = False,
) -> None:
	"""
	Plot statistics from the dataset.
	This command computes the statistics of the dataset and plots the numerosity of each throw.

	:param output: Path to save the output plot image; if not set, the plot will be displayed on the screen.
	:param stat: Statistic to plot, 'throw' for numerosity, 'duration' for duration.
	:param sort: Sorting parameter:
	:param grouped: Whether to apply aliases and group similar throws together.
	:return: ``None``
	"""

	# Initialize dataset
	handler = DatasetOrchestrator()

	# Load all the data
	handler.load_all(MyEnv.dataset_source)
	handler.finalize(group_throws=grouped)

	# Compute statistics
	stats = handler.compute_stats()

	# Sort values
	throws = stats[stat if stat != 'average' else 'duration']
	throws = sorted(throws.items(), key=lambda item: item[sort], reverse=bool(sort))

	if stat == 'duration':
		# Convert duration from [HH:]MM:SS.000 to seconds (and decimal milliseconds)
		throws = [(key, ts_to_sec(value)) for key, value in throws]
	elif stat == 'average':
		# Convert average duration from [HH:]MM:SS.000 to seconds (and decimal milliseconds)
		throws = [(key, ts_to_sec(value, default_ms=0)) for key, value in throws]
		# Get throws numerosity
		n_throws = stats['throw']
		# Average duration
		throws = [(key, value / n_throws[key]) for key, value in throws]

	# Split keys and values
	throw_keys, throw_values = zip(*throws)

	# Change names to Title Case
	throw_keys = [key.replace('_', ' ').title() for key in throw_keys]

	# Create plot
	fix, ax = plt.subplots(figsize=(10, 6))

	# Normalize values for color mapping
	max_value = max(throw_values) if throw_values else 1
	# noinspection PyUnresolvedReferences
	colors = plt.cm.viridis(np.array(throw_values) / max_value)

	# Create the horizontal bar plot.
	ax.barh(throw_keys[::-1], throw_values[::-1], color=colors[::-1])

	# Add labels and a title for clarity
	ax.set_xlabel(stat_keys[stat])
	ax.set_ylabel('Throw')
	ax.set_title('Throws Statistics')

	# Add the value label at the end of each bar for better readability
	for index, value in enumerate(throw_values[::-1]):
		ax.text(value, index, f' {value:.4}', va='center')

	# Adjust layout to prevent labels from being cut off
	plt.tight_layout()

	# Save or show the plot
	if output:
		plt.savefig(output, dpi=400, bbox_inches='tight')
		print(f"Saved plot to {output}")
	else:
		plt.show()

	return


@cmd_plot.command(name='histogram')
@click.option(
	'--input', '-i', 'input_',
	required=True,
	help='Path to the input image or the numpy file containing histogram data.'
)
@click.option(
	'--output', '-o',
	default=None,
	help='Path to save the output plot image. If not set, the plot will be displayed on the screen.'
)
def cmd_plot_histogram(
		input_: str,
		output: str | None = None,
) -> None:
	"""
	Create Hue/Value colour profile histograms from ROI images.
	The input can either be an image file, or a numpy file containing
	the already-computed Hue/Value histogram data.
	If the output is not set, the plot will be displayed on the screen.

	:param input_: Path to the input image or the numpy file.
	:param output: Path to save the output plot image.
	:return: ``None``
	"""

	# Check file exists
	if not os.path.exists(input_):
		raise ValueError(f"Input file does not exist: {input_}")

	if input_.endswith('.npy') or input_.endswith('.bin'):

		# Read binary file
		hist = np.load(input_)
		if hist.size == 0:
			raise ValueError(f"Error reading binary file: {input_}")

	else:

		# Open image
		image = cv2.imread(input_)
		if image is None:
			raise ValueError(f"Error opening image file: {input_}")

		# Compute histogram
		fb = FrameBox(0, 0, image.shape[1], image.shape[0])
		hist = fb.calc_hv_histogram(image, normalize=True)

	# Reduce histogram by 4
	reduce_factor = 1
	hist = hist.reshape(hist.shape[0] // reduce_factor, reduce_factor, hist.shape[1] // reduce_factor, reduce_factor)
	hist = hist.sum(axis=(1, 3))
	cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

	# Create plot
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	# Map histogram data to 3D bar plot
	xx, yy = np.meshgrid(
		np.arange(hist.shape[0]),  # Hue bins
		np.arange(hist.shape[1])  # Value bins
	)
	x_flat = xx.flatten()
	y_flat = yy.flatten()
	dz_flat = hist.flatten()  # These are the heights of the bars

	colormap = plt.get_cmap('viridis')
	colors = colormap(dz_flat)

	# Plot the histogram
	ax.bar3d(
		x=x_flat,
		y=y_flat,
		z=0,
		dx=1,
		dy=1,
		dz=dz_flat,
		color=colors,
		shade=True,
		axlim_clip=True,
	)
	ax.set_xlabel('Hue')
	ax.set_ylabel('Value')
	ax.set_zlabel('Frequency')

	# Save or show the plot
	if output:
		plt.savefig(output, dpi=400, bbox_inches='tight')
		print(f"Saved plot to {output}")
	else:
		plt.show()

	return
