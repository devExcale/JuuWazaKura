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
@click.option(
	'--model', '-m',
	default=None,
	help=f'Model to train: {JwkModel.models.keys()}'
)
def cmd_train(epochs: int, model: str) -> None:
	"""
	Train the model.

	:param epochs: Number of epochs to train the model
	:param model: Model type
	"""

	handler = JwkModel(model)
	handler.fit_model(epochs=epochs)
	handler.save_weights("model.tf")

	return


@cmd_model.command(name='preprocess')
@click.option(
	'--input', '-i', 'input_',
	default='*',
	help='Path to the input folder containing the videos to preprocess'
)
@click.option(
	'--output', '-o',
	default='pre',
	help='Path to the output folder'
)
def cmd_preprocess(input_: str, output: str) -> None:
	"""
	Preprocess the video files.

	:param input_: Path to the input folder containing the videos to preprocess
	:param output: Path to the output folder
	"""

	if not input_:
		raise ValueError('Input path is required.')

	if input_ == '*':
		sets = {
			folder
			for folder in os.listdir(MyEnv.dataset_segments)
			if os.path.isdir(os.path.join(MyEnv.dataset_segments, folder))
		}
	else:
		sets = {input_}

	pathjoin = os.path.join

	videos = {
		pathjoin(MyEnv.dataset_segments, folder, filename): pathjoin(MyEnv.dataset_inputready, folder, filename)
		for folder in sets
		for filename in os.listdir(os.path.join(MyEnv.dataset_segments, folder))
		if filename.endswith('.mp4')
	}

	preprocessor = JwkPreprocessor()

	with click.progressbar(
			videos.items(),
			label='Preprocessing...',
			show_eta=False,
			show_percent=True,
			item_show_func=lambda t: t[0].split('/')[-1].split('\\')[-1] if t else '',
	) as bar:

		# Iterate over the videos
		for vid_in, vid_out in bar:

			# Check if the output video already exists
			if os.path.exists(vid_out):
				continue

			# Create output directory if it doesn't exist
			os.makedirs(os.path.dirname(vid_out), exist_ok=True)

			# Preprocess the video
			preprocessor.preprocess_video(vid_in, vid_out)

	return


if __name__ == '__main__':
	cmd_model()
