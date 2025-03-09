import json
import logging
import os
import threading
from queue import Queue

from yt_dlp import YoutubeDL

from .ds_instance import DatasetInstance
from ..clipper import Clipper
from ..utils import MyEnv, get_logger

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())


class DatasetDownloader:

	def __init__(self, dir_dataset: str, dir_segments: str) -> None:
		"""
		Constructor for the DatasetDownloader class.
		"""

		self.dir_dataset: str = dir_dataset
		""" Directory where the dataset is stored """

		self.dir_segments: str = dir_segments
		""" Directory where the segments are stored """

		self.rm_clip_source: bool = MyEnv.delete_yt
		""" Whether to remove the source video after clipping """

		self.yt_format: dict[str, int | str] = {}
		""" YouTube video format (width, height, fps, ext), loaded on enter """

		self.yt_video_ids: dict[str, str] = {}
		""" YouTube video Name:IDs, loaded on enter """

		self.queue_yt: Queue[DatasetInstance | None] = Queue()
		""" Queue for YouTube consumer """

		self.queue_clip: Queue[DatasetInstance | None] = Queue()
		""" Queue for Clipper consumer """

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
		self.yt_video_ids = {
			k: v
			for k, v in ds['vidIds'].items()
			if not MyEnv.dataset_include or k in MyEnv.dataset_include
		}

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
			log.warning(f'Skipping {name} as the corresponding segments do not exist')
			return False

		# Check whether the video already exists
		path_video = os.path.join(self.dir_segments, f'{name}.mp4')
		if os.path.exists(path_video):
			log.info(f'Skipping {name} as the video already exists')
			return True

		# Create the yt-dlp options
		ydl_opts = {
			'outtmpl': os.path.join(self.dir_segments, f'{name}.%(ext)s'),
			'format': self.yt_find_format(yt_id),
			'quiet': True,
			'noprogress': True,
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

	def segment_video(self, name: str) -> None:
		"""
		Segment a video given its name.

		:param name: Filename of the video (no extension)
		"""

		# Check whether the video exists
		path_video = os.path.join(self.dir_segments, f'{name}.mp4')
		if not os.path.exists(path_video):
			log.warning(f'Skipping {name} as the video does not exist')
			return

		# Segment the video
		clipper = Clipper(
			input_filepath=path_video,
			savedir=self.dir_segments,
			name=name,
		)

		with clipper as c:
			c.export_from_csv(os.path.join(self.dir_dataset, f'{name}.csv'))

		# Delete file on finish
		if self.rm_clip_source:
			os.remove(path_video)

		return

	def download_segment_all_sync(self):
		"""
		Download and segment all the videos in the dataset, synchronously.

		:return:
		"""

		log.info('Downloading dataset synchronously')

		# Create the directory for the videos
		if not os.path.exists(self.dir_segments):
			os.makedirs(self.dir_segments)

		log.debug(f'Downloading dataset to "{self.dir_segments}"')

		todo_yt = []
		todo_clip = []

		# Add all the datasets to the queue
		for competition in self.yt_video_ids:
			try:

				ds = DatasetInstance(
					competition=competition,
					dir_dataset=self.dir_dataset,
					dir_segments=self.dir_segments,
				)
				ds.validate_dataset()
				todo_yt.append(ds)

			except Exception as e:
				log.error(f'Cannot load {competition}: {e}', exc_info=e)
				continue

		# Number of videos to download
		n_todownload = len(todo_yt)

		# Download videos from YouTube
		for i, ds in enumerate(todo_yt):
			# Human-readable index
			i = i + 1
			try:

				# Get YouTube id
				yt_id = self.yt_video_ids[ds.competition]

				# Skip download: no csv
				if not ds.is_csv_present():
					log.warning(f'(YouTube {i}/{n_todownload}) {ds.competition}: Skipping, CSV not found')
					continue

				# Skip download: video present or segments present (send to clipper)
				if ds.is_video_present() or not ds.missing_segments():
					log.debug(f'(YouTube {i}/{n_todownload}) {ds.competition}: Skipping, video or all segments found')
					todo_clip.append(ds)
					continue

				log.info(f'(YouTube {i}/{n_todownload}) {ds.competition}: Downloading... ({yt_id})')

				# Download video and send to clipper
				self.yt_download(ds.competition, yt_id)
				todo_clip.append(ds)

				log.info(f'(YouTube {i}/{n_todownload}) {ds.competition}: Downloading done ({yt_id})')

			except Exception as e:
				log.error(f'(YouTube {i}/{n_todownload}) {ds.competition}: {e}', exc_info=e)

		# Number of downloaded videos
		n_toclip = len(todo_clip)

		# Convert full videos to segments
		for i, ds in enumerate(todo_clip):
			# Human-readable index
			i = i + 1
			try:

				# Skip segment: no csv
				if not ds.is_csv_present():
					log.warning(f'(Clipper {i}/{n_toclip}) {ds.competition}: Skipping, CSV not found')
					continue

				# Skip segment: segments present
				if not ds.missing_segments():
					log.debug(f'(Clipper {i}/{n_toclip}) {ds.competition}: Skipping, all segments found')
					continue

				log.info(f'(Clipper {i}/{n_toclip}) {ds.competition}: Segmenting...')

				# Segment the video
				self.segment_video(ds.competition)

				log.info(f'(Clipper {i}/{n_toclip}) {ds.competition}: Segmenting done')

			except Exception as e:
				log.error(f'(Clipper {i}/{n_toclip}) {ds.competition}: {e}', exc_info=e)

		log.info('Finished downloading and segmenting all videos')

		return

	def download_segment_all_async(self):

		log.info('Downloading dataset asynchronously')

		# Start the YouTube thread
		thread_yt = threading.Thread(target=self.__consumer_yt__)
		thread_yt.start()

		# Start the Clipper consumers
		n_consumers = MyEnv.concurrent_clippers
		consumers = []
		for _ in range(n_consumers):
			consumer = threading.Thread(target=self.__consumer_clipper__)
			consumer.start()
			consumers.append(consumer)

		# Add all the datasets to the queue
		for competition in self.yt_video_ids:
			try:

				ds = DatasetInstance(
					competition=competition,
					dir_dataset=self.dir_dataset,
					dir_segments=self.dir_segments,
				)
				ds.validate_dataset()
				self.queue_yt.put(ds)

			except Exception as e:
				log.error(f'Cannot load {competition}: {e}', exc_info=e)
				continue

		# End the queue
		self.queue_yt.put(None)

		# Wait for the thread_yt to finish
		thread_yt.join()

		# Wait for the consumers to finish
		for consumer in consumers:
			consumer.join()

		log.info('Finished downloading and segmenting all videos')

	def __consumer_yt__(self) -> None:

		while True:

			# Get dataset instance
			ds = self.queue_yt.get()

			# Check queue end
			if ds is None:
				self.queue_yt.put(None)
				break

			try:

				# Get YouTube id
				yt_id = self.yt_video_ids[ds.competition]

				# Skip download: no csv
				if not ds.is_csv_present():
					log.warning(f'(YouTube Consumer) {ds.competition}: Skipping, CSV not found')
					self.queue_yt.task_done()
					continue

				# Skip download: video present or segments present (send to clipper)
				if ds.is_video_present() or not ds.missing_segments():
					log.debug(f'(YouTube Consumer) {ds.competition}: Skipping, video or all segments found')
					self.queue_clip.put(ds)
					self.queue_yt.task_done()
					continue

				log.info(f'(YouTube Consumer) {ds.competition}: Downloading... ({yt_id})')

				# Download video and send to clipper
				self.yt_download(ds.competition, yt_id)
				self.queue_clip.put(ds)
				self.queue_yt.task_done()

				log.info(f'(YouTube Consumer) {ds.competition}: Downloading done ({yt_id})')

			except Exception as e:
				self.queue_yt.task_done()
				log.error(f'(YouTube Consumer) {ds.competition}: {e}', exc_info=e)

		# Signal the clipper consumer there are no more videos
		self.queue_clip.put(None)

		return

	def __consumer_clipper__(self):

		while True:

			# Get the dataset instance
			ds = self.queue_clip.get()

			# Check if the producer is done
			if ds is None:
				self.queue_clip.put(None)
				break

			try:

				# Skip segment: no csv
				if not ds.is_csv_present():
					log.warning(f'(Clipper Consumer) {ds.competition}: Skipping, CSV not found')
					self.queue_clip.task_done()
					continue

				# Skip segment: segments present
				if not ds.missing_segments():
					log.debug(f'(Clipper Consumer) {ds.competition}: Skipping, all segments found')
					self.queue_clip.task_done()
					continue

				log.info(f'(Clipper Consumer) {ds.competition}: Segmenting...')

				# Segment the video
				self.segment_video(ds.competition)
				self.queue_clip.task_done()

				log.info(f'(Clipper Consumer) {ds.competition}: Segmenting done')

			except Exception as e:
				self.queue_clip.task_done()
				log.error(f'(Clipper Consumer) {ds.competition}: {e}', exc_info=e)

		return


def padded_ratio(n: int, top: int, pad: str = " ") -> str:
	digits = lambda x: len(str(x))
	return pad * (digits(top) - digits(n)) + str(n) + '/' + str(top)
