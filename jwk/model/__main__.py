import logging
import os

import click

from .model import JwkModel
from .preprocessor import JwkPreprocessor
from ..utils import get_logger, MyEnv

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())


@click.group(name='model')
def cmd_model() -> None:
	"""
	Model command line interface.
	"""

	return


@cmd_model.command(name='train')
@click.option(
	'--epochs', '-e',
	default=10,
	help='Number of epochs to train the model'
)
def cmd_train(epochs: int) -> None:
	"""
	Train the model.

	:param epochs: Number of epochs to train the model
	"""

	handler = JwkModel()
	handler.fit_model(epochs=epochs)
	handler.save_weights("model.weights.h5")

	return


@cmd_model.command(name='preprocess')
@click.option(
	'--input', '-i', 'input_',
	default=None,
	help='Path to the input folder containing the videos to preprocess'
)
@click.option(
	'--output', '-o',
	default=None,
	help='Path to the output folder'
)
def cmd_preprocess(input_: str, output: str) -> None:
	"""
	Preprocess the video files.

	:param input_: Path to the input folder containing the videos to preprocess
	:param output: Path to the output folder
	"""

	if not input_ or not output:
		raise ValueError('Input and output paths are required.')

	folder_in = os.path.join(MyEnv.dataset_clips, input_)
	folder_out = os.path.join(MyEnv.dataset_source, output)

	preprocessor = JwkPreprocessor()

	videos = [filename for filename in os.listdir(folder_in) if filename.endswith('.mp4')]

	with click.progressbar(
			videos,
			label='Preprocessing...',
			show_eta=False,
			show_percent=True,
			item_show_func=lambda s: s if s else '',
	) as bar:

		# Iterate over the videos
		for filename in bar:
			# Get the input and output paths
			filepath_in = os.path.join(folder_in, filename)
			filepath_out = os.path.join(folder_out, filename)

			# Create output directory if it doesn't exist
			os.makedirs(os.path.dirname(filepath_out), exist_ok=True)

			# Preprocess the video
			preprocessor.preprocess_video(filepath_in, filepath_out)

	return


if __name__ == '__main__':
	cmd_model()
