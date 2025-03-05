import json
import logging
import os
import queue
import threading

from yt_dlp import YoutubeDL

from ..clipper import Clipper
from ..utils import MyEnv, get_logger

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())


class DatasetDownloader:

	def __init__(self, dir_dataset: str) -> None:
		"""
		Constructor for the DatasetDownloader class.

		:param dir_dataset: Directory where the dataset is stored
		"""

		self.dir_dataset: str = dir_dataset
		""" Directory where the dataset is stored """

		self.dir_clips: str = os.path.join(self.dir_dataset, 'clips')
		""" Directory where the clips are stored """

		self.yt_format: dict[str, int | str] = {}
		""" YouTube video format (width, height, fps, ext), loaded on enter """

		self.yt_video_ids: dict[str, str] = {}
		""" YouTube video Name:IDs, loaded on enter """

		self.queue = queue.Queue()
		""" Multi-threading queue """

		return

	def __enter__(self) -> "DatasetDownloader":
		"""

		:return:
		"""

		# Ensure dataset directory exists
		if not os.path.exists(self.dir_dataset):
			raise FileNotFoundError(f'Dataset directory not found: {self.dir_dataset}')

		log.debug(f'Dataset directory: {self.dir_dataset}')

		# Read dataset json file
		with open(os.path.join(self.dir_dataset, 'dataset.json')) as f:
			ds = json.load(f)

		# Get dataset parameters
		self.yt_format = ds['format']
		self.yt_video_ids = ds['vidIds']

		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:
		return

	def yt_download(self, name: str, yt_id: str) -> bool:
		"""
		Download a video from YouTube given its ID.

		:param name: Filename of the video (no extension)
		:param yt_id: ID of the video
		:return: True if the video was downloaded, False otherwise
		"""

		# Check whether a corresponding csv exists
		path_csv = os.path.join(self.dir_dataset, f'{name}.csv')
		if not os.path.exists(path_csv):
			log.warning(f'Skipping {name} as the corresponding clippings do not exist')
			return False

		# Check whether the video already exists
		path_video = os.path.join(self.dir_clips, f'{name}.mp4')
		if os.path.exists(path_video):
			log.info(f'Skipping {name} as the video already exists')
			return True

		# Create the yt-dlp options
		ydl_opts = {
			'outtmpl': os.path.join(self.dir_clips, f'{name}.%(ext)s'),
			'format': self.yt_find_format(yt_id),
			'quiet': True,
			'concurrent_fragment_downloads': MyEnv.concurrent_downloads,
		}

		# Download the file
		with YoutubeDL(ydl_opts) as ydl:
			ret = ydl.download([yt_id])

		return ret == 0

	def yt_find_format(self, yt_id) -> str:

		# Create the yt-dlp options
		ydl_opts = {
			'quiet': True,
		}

		# Download video info
		with YoutubeDL(ydl_opts) as ydl:
			info = ydl.extract_info(yt_id, download=False)

		# Get the formats
		formats: list[dict] = info['formats']

		# Filter by the desired format
		formats = filter(
			lambda f: all(f.get(k, None) == v for k, v in self.yt_format.items()),
			formats
		)

		if not formats:
			raise ValueError(f'No format found for {yt_id}')

		# Get format with lowest filesize
		fmt = min(formats, key=lambda f: f.get('filesize', float('inf')))

		# Return format id
		return fmt['format_id']

	def clip_video(self, name: str) -> None:
		"""
		Clip a video given its name.

		:param name: Filename of the video (no extension)
		"""

		# Check whether the video exists
		path_video = os.path.join(self.dir_dataset, 'clips', f'{name}.mp4')
		if not os.path.exists(path_video):
			log.warning(f'Skipping {name} as the video does not exist')
			return

		# Clip the video
		clipper = Clipper(
			input_filepath=path_video,
			savedir=os.path.join(self.dir_dataset, 'clips'),
			name=name,
		)

		with clipper as c:
			c.export_from_csv(os.path.join(self.dir_dataset, f'{name}.csv'))

		return

	def main_dwnl_clip_sync(self):
		"""
		Download and clip all the videos in the dataset, synchronously.

		:return:
		"""

		# Create the directory for the videos
		if not os.path.exists(self.dir_clips):
			os.makedirs(self.dir_clips)

		log.debug(f'Downloading dataset to "{self.dir_clips}"')

		n_vids = len(self.yt_video_ids)
		downloaded = []

		# Download all the files
		for i, (name, yt_id) in enumerate(self.yt_video_ids.items()):

			log.info(f'Downloading {i + 1}{n_vids}: {name}/{yt_id}')

			if self.yt_download(name, yt_id):
				downloaded.append(name)

		n_downloaded = len(downloaded)

		log.info(f'Downloaded {n_downloaded}/{n_vids} videos')

		# Convert full videos to clips
		for i, name in enumerate(downloaded):
			log.info(f'Clipping {i + 1}/{n_downloaded}: {name}')

			clipper = Clipper(
				input_filepath=os.path.join(self.dir_clips, f'{name}.mp4'),
				savedir=self.dir_clips,
				name=name,
			)

			with clipper as c:
				c.export_from_csv(os.path.join(self.dir_dataset, f'{name}.csv'))

		log.info('Finished downloading and clipping all videos')

		return

	def main_dwnl_clip_async(self):

		# Start the producer
		producer = threading.Thread(target=self.__yt_producer__)
		producer.start()

		# Start the consumers
		n_consumers = MyEnv.concurrent_clippers
		consumers = []
		for _ in range(n_consumers):
			consumer = threading.Thread(target=self.__clip_consumer__)
			consumer.start()
			consumers.append(consumer)

		# Wait for the producer to finish
		producer.join()

		# Wait for the consumers to finish
		for consumer in consumers:
			consumer.join()

		log.info('Finished downloading and clipping all videos')

	def __yt_producer__(self):

		n_vids = len(self.yt_video_ids)

		# Download all the files
		for i, (name, yt_id) in enumerate(self.yt_video_ids.items()):

			log.info(f'Downloading {i + 1}{n_vids}: {name}/{yt_id}')

			if self.yt_download(name, yt_id):
				self.queue.put(name)

		# Signal the consumer that the producer is done
		self.queue.put(None)

	def __clip_consumer__(self):

		while True:

			# Get the name of the downloaded video
			name = self.queue.get()

			# Check if the producer is done
			if name is None:
				self.queue.put(None)
				break

			# Clip the video
			self.clip_video(name)
			self.queue.task_done()


def padded_ratio(n: int, top: int, pad: str = " ") -> str:
	digits = lambda x: len(str(x))
	return pad * (digits(top) - digits(n)) + str(n) + '/' + str(top)
