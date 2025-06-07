import logging
import os
from typing import Union, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

from ..utils import get_framestamp, ts_to_sec

log = logging.getLogger(__name__)


class Clipper:
	"""
	Class for generating clips from a video file.
	"""

	def __init__(
			self,
			path_source: str,
			dir_output: str,
			name: str,
			size: Optional[Tuple[int, int]] = None,
			default_ms_end: int = 0,
	) -> None:
		"""
		Initializes the Clipper object with the video file path.

		:param path_source: Path to the input video file.
		:param dir_output: Directory where the output clips will be saved.
		:param name: Prefix for the output clips.
		:param size: Size of the output clips as a tuple (width, height).
			If ``None``, the clips will be saved in the original size of the video.
		:param default_ms_end: Default milliseconds to add to the end timestamp if no milliseconds are provided.
		"""

		self.path_source = path_source
		""" Path to the input video file. """

		self.dir_output = dir_output
		""" Directory where the output clips will be saved. """

		self.name = name
		""" Prefix for the output clips. """

		self.output_size = size
		""" Size of the output clips as a tuple (width, height). """

		self.default_ms_end = default_ms_end
		""" Default milliseconds to add to the end timestamp if no milliseconds are provided. """

		self.cap: cv2.VideoCapture | None = None
		""" Video capture object for reading the video file. """

		self.frame_size: tuple[int, int] | None = None
		""" Size of the video frames as a tuple (width, height). """

		self.frame_count: int = 0
		""" Total number of frames in the video. """

		self.fps: float = 0.0
		""" Frames per second of the video. """

		return

	def __enter__(self) -> 'Clipper':
		"""
		Opens the video file and initializes the video capture object.

		:return: ``self``
		"""

		# Open video file
		cap = cv2.VideoCapture(self.path_source)

		# Test if the video file is opened successfully
		if not cap.isOpened():
			raise FileNotFoundError(f"Could not open video file at path: {self.path_source}")

		self.cap = cap

		# Get video properties
		self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		self.fps = cap.get(cv2.CAP_PROP_FPS)

		# Set the output size
		if self.output_size is not None and len(self.output_size) == 2:
			w, h = self.output_size

			if w is None and h is None:
				self.output_size = None
			elif h is not None:
				self.output_size = (int(h * self.frame_size[0] / self.frame_size[1]), h)
			elif w is not None:
				self.output_size = (w, int(w * self.frame_size[1] / self.frame_size[0]))

		log.debug(f"Output size: {self.output_size}")

		# Create the output directory
		if not os.path.exists(self.dir_output):
			os.makedirs(self.dir_output)

		log.debug(f"Output directory: {self.dir_output}")

		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:
		"""
		Releases the video capture object.
		"""
		self.cap.release()

		return

	def export(self, start: Union[int, str], end: Union[int, str], sim: bool = False) -> None:
		"""
		Exports a clip from the opened video to another file.

		:param start: Timestamp in the format `[HH:]MM:SS.000` or milliseconds from the start of the video
		:param end: Timestamp in the format `[HH:]MM:SS.000` or milliseconds from the start of the video
		:param sim: Simulate exporting the clip without actually writing the file
		"""

		# Convert timestamps to milliseconds and frame numbers
		ms_start = int(ts_to_sec(start) * 1000)
		ms_end = int(ts_to_sec(end, self.default_ms_end) * 1000)
		frame_start, frame_end = get_framestamp(start, end, self.fps, self.default_ms_end)

		# Check if the start and end frames are within the video length
		if frame_start < 0 or frame_end < 0 or frame_start >= self.frame_count or frame_end >= self.frame_count:
			raise ValueError(f"Start and end frames must be within the video length: {self.name}/{ms_start}-{ms_end}")

		# Check if start/end are in the correct order
		if frame_start > frame_end:
			raise ValueError(f"Start frame must be before the end frame: {self.name}/{ms_start}-{ms_end}")

		clip_name = f"{self.name}-{ms_start}.mp4"
		clip_path = os.path.join(self.dir_output, clip_name)

		# Check if clip was already exported
		if os.path.exists(clip_path):
			log.debug(f"Clip {clip_name} already exists")
			return
		else:
			log.debug(f"Exporting {clip_name}")

		if sim:
			return

		# Set the start frame
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

		# Initialize the video writer
		out_frame_size = self.output_size or self.frame_size
		# noinspection PyUnresolvedReferences
		writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'avc1'), self.fps, out_frame_size)

		# Read and write the frames to the output video file
		for i in range(frame_start, frame_end + 1):

			ok, frame = self.cap.read()
			if not ok:
				break

			# Sharpen the frame
			# frame = sharpen(frame)

			# Resize the frame
			if self.output_size is not None:
				frame = cv2.resize(frame, self.output_size)

			writer.write(frame)

		# Release the video writer
		writer.release()

		return

	# noinspection PyUnresolvedReferences
	def export_from_csv(self, csv_path: str, sim: bool = False) -> None:
		"""
		Exports clips from the opened video based on the timestamps in a CSV file.
		The CSV file is required to have a heading row with the columns `ts_start` and `ts_end`,
		the timestamps must be in the format `[HH:]MM:SS`.

		:param csv_path: Path to the CSV file containing the timestamps
		:param sim: Simulate exporting the clips without actually writing the files
		"""

		# Read the CSV file
		df = pd.read_csv(csv_path, sep=',', header=0)

		# Export clips based on the timestamps
		for row in df.itertuples():
			ts_start = row.ts_start
			ts_end = row.ts_end

			self.export(ts_start, ts_end, sim=sim)

		return


def sharpen(image: np.ndarray) -> np.ndarray:
	# Apply the unsharp mask
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	return cv2.filter2D(image, -1, kernel)
