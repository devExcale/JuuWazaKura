import logging
import os.path
import typing
from typing import Dict, Any, List

from dotenv import load_dotenv

log = logging.getLogger(__name__)


class MyEnv:
	"""
	Class to manage the environment variables.
	Tries to load the class variables from the system environment.
	Class variables are recognized as such if they are annotated by their type.
	"""

	concurrent_downloads: int = 8
	"""How many fragments to download concurrently, a yt-dlp parameter."""

	concurrent_clippers: int = 4
	"""How many videos to clip concurrently."""

	dataset_source: str = os.path.join(os.getcwd(), 'dataset')
	"""Path to the folder containing the csv files."""

	dataset_clips: str = os.path.join(os.getcwd(), 'dataset', 'clips')
	"""Path to the folder containing the video clips."""

	dataset_include: List[str] = []
	""" Subset of the dataset to include (dataset id). Empty list includes all. """

	delete_yt: bool = True
	"""Whether to delete the original dataset videos after clipping."""

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
			cast = cls.__annotations__.get(key, str)

			# Custom converters
			if list in (cast, typing.get_origin(cast)):
				cast = lambda x: list(filter(len, str(x).split(',')))

			# Get the value from the environment
			value = os.getenv(key)

			# Set the casted value if present
			if value is not None:

				try:
					value = cast(value)
				except ValueError as e:
					log.error(f'Could not cast \'{key}={value}\' to {cast}: {e}')
					continue

				setattr(cls, key, value)


MyEnv.apply_dotenv()
