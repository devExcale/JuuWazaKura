import logging
import os.path
import typing
from typing import Dict, Any, List

from dotenv import load_dotenv

from .logging import get_logger

log = get_logger(__name__, logging.ERROR)

true_values = ['true', '1', 'yes']
false_values = ['false', '0', 'no']


class MyEnv:
	"""
	Class to manage the environment variables.
	Tries to load the class variables from the system environment.
	Class variables are recognized as such if they are annotated by their type.
	"""

	concurrent_downloads: int = 8
	""" How many fragments to download concurrently, a yt-dlp parameter. """

	concurrent_clippers: int = 4
	""" How many videos to clip concurrently. """

	dataset_source: str = os.path.join(os.getcwd(), 'dataset')
	""" Path to the folder containing the csv files. """

	dataset_clips: str = os.path.join(os.getcwd(), 'dataset', 'clips')
	""" Path to the folder containing the video clips. """

	dataset_preproc: str = os.path.join(os.getcwd(), 'dataset', 'pre')
	""" Path to the folder containing the video clips. """

	dataset_windows: str = os.path.join(os.getcwd(), 'dataset', 'win')
	""" Path to the folder containing the video clips. """

	dataset_include: List[str] = []
	""" Subset of the dataset to include (dataset id). Empty list includes all. """

	dataset_exclude: List[str] = []
	""" Subset of the dataset to include (dataset id). Empty list includes all. """

	delete_yt: bool = True
	""" Whether to delete the original dataset videos after clipping. """

	log_levelname: str = 'INFO'
	""" The logging level name. """

	yolo_model: str = 'yolo11s.pt'
	""" Path to the YOLO model to use for pre-processing. """

	preprocess_n_ymax: int = 5
	""" Number of boxes lowest in the frame to keep during preprocessing, 0 to keep all. """

	segments_per_batch: int = 1
	""" Number of segments to load in a single batch, not including transformations. """

	tf_dump_debug_info: bool = False
	""" Whether to enable TensorFlow ``tf.debugging.experimental.enable_dump_debug_info()``. """

	@classmethod
	def log_level(cls) -> int:
		"""
		Get the logging level from the class variable.
		"""

		return getattr(logging, cls.log_levelname, logging.INFO)

	@classmethod
	def get_keys(cls) -> List[str]:
		"""
		Get the keys of the class variables.
		"""

		keys = [
			k
			for k in cls.__annotations__.keys()
			if not k.startswith('_')
				and not callable(getattr(cls, k))
		]

		return keys

	@classmethod
	def values(cls) -> Dict[str, Any]:
		"""
		Get the values of the class variables.
		"""

		items = {
			key: getattr(cls, key)
			for key in cls.get_keys()
		}

		return items

	@classmethod
	def apply_dotenv(cls) -> None:
		"""
		Load the .env file into the environment and set the class variables.
		"""

		load_dotenv()

		for key in cls.get_keys():

			# Get annotated type for the variable
			clazz = cls.__annotations__.get(key, str)
			cast = clazz

			# Custom converters
			if bool is cast:
				cast = str_to_bool
			if list in (cast, typing.get_origin(cast)):
				cast = lambda x: list(filter(len, str(x).split(',')))

			# Get the value from the environment
			value = os.getenv(key)

			# Set the casted value if present
			if value is not None:

				try:
					value = cast(value)
				except ValueError as e:
					log.error(f'Could not cast \'{key}={value}\' to {clazz}: {e}')
					continue

				setattr(cls, key, value)


def str_to_bool(val: str) -> bool:
	"""
	Convert a string to a boolean, case insensitive.

	:param val: The string to convert
	:return: The boolean value
	"""

	if val.lower() in true_values:
		return True
	elif val.lower() in false_values:
		return False

	raise ValueError(f'Cannot convert \'{val}\' to bool')


MyEnv.apply_dotenv()
