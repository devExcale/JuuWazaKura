import os

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..model import FrameBox


@click.command(name='plot')
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
def cmd_plot(
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
