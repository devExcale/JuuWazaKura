import argparse
import json
from typing import Optional

from ..commons import filename_from_path

parser = argparse.ArgumentParser(
	description='Fit a skeleton to two judo athletes in a video file.'
)

parser.add_argument(
	'--input', '-i',
	dest='input_video',
	type=str,
	help='Path to the input video file',
	required=True,
)
parser.add_argument(
	'-o',
	dest='save_dir',
	type=str,
	help='Output directory, defaults to the current directory',
	required=False,
)
parser.add_argument(
	'--name', '-n',
	dest='name',
	type=str,
	help='Name of the export file, defaults to fit-{filename}',
	required=False,
)
parser.add_argument(
	'--preview', '-p',
	dest='export_preview',
	action='store_true',
	help='Whether to save a resulting video with the pose fitting',
	required=False,
)
parser.add_argument(
	'--export', '-e',
	dest='export_results',
	action='store_true',
	help='Whether to export the resulting fits to a CSV file',
	required=False,
)
parser.add_argument(
	'--model', '-m',
	dest='model_type',
	type=str,
	help='Type of the model to use for pose fitting (s/x), defaults to s',
	required=False,
	default='s',
)


class PoseFitterArguments:

	def __init__(self):

		self.input_video: Optional[str] = None
		"Filepath to the input video file."

		self.save_dir: Optional[str] = None
		"Output directory for the results."

		self.name: Optional[str] = None
		"Name of the output file, without extension."

		self.export_preview: bool = False
		"Whether to save a resulting video with the pose fitting."

		self.export_results: bool = True
		"Whether to export the resulting fits to a CSV file."

		self.model_type: str = 's'
		"Type of the model to use for pose fitting (s/x)."

	def __computed_defaults__(self):
		"""
		Compute the fields that have default values depending on other fields.
		"""

		# Set default output directory
		if self.save_dir is None:
			self.save_dir = '.'

		# Set default output name
		if self.name is None and self.input_video is not None:
			self.name = filename_from_path(self.input_video)

	def from_parser(self) -> 'PoseFitterArguments':
		"""
		Parse the arguments from the command line.
		:return: The PoseFitterArguments object
		"""

		# Get input arguments
		args = parser.parse_args()

		# Set the arguments
		self.input_video = args.input_video
		self.save_dir = args.save_dir
		self.name = args.name
		self.export_preview = args.export_preview
		self.export_results = args.export_results

		# Set default values
		self.__computed_defaults__()

		return self

	def validity_barrier(self):
		"""
		Check whether the current configuration is valid.
		If not, raise an exception with the reason.
		"""

		if self.input_video is None:
			raise ValueError("No input video file provided.")

		if self.save_dir is None:
			raise ValueError("No output directory provided.")

		if self.name is None:
			raise ValueError("No output name provided.")

		if not self.export_preview and not self.export_results:
			raise ValueError("One of the export options must be enabled.")

		if self.model_type not in ['s', 'x']:
			raise ValueError("Model type must be either 's' or 'x'.")

	def __str__(self) -> str:
		return json.dumps(self.__dict__)
