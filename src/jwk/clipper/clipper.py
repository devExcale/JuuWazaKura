import logging
import os
from typing import Union, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

from jwk.clipcommons import ClipCommons

log = logging.getLogger(__name__)


class Clipper(ClipCommons):
	"""
	Class for generating clips from a video file.
	"""

	def __init__(self, input_filepath: str, savedir: str, name: str, size: Optional[Tuple[int, int]] = None) -> None:
		"""
		Initializes the Clipper object with the video file path.
		"""
		self.input_filepath = input_filepath
		self.root_dir = savedir
		self.name = name
		self.output_size = size
		self.output_dir = os.path.join(self.root_dir, self.name)

		return

	def __enter__(self) -> 'Clipper':
		# Open video file
		cap = cv2.VideoCapture(self.input_filepath)

		# Test if the video file is opened successfully
		if not cap.isOpened():
			raise FileNotFoundError(f"Could not open video file at path: {self.input_filepath}")

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
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		log.debug(f"Output directory: {self.output_dir}")

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

		:param start: Start frame number or timestamp in the format `[HH:]MM:SS`
		:param end: End frame number or timestamp in the format `[HH:]MM:SS`
		:param sim: Simulate exporting the clip without actually writing the file
		"""

		# Convert timestamps to frame numbers
		start, end = self.__ensure_timestamps__(start, end)

		# Check if the start and end frames are within the video length
		if start < 0 or end < 0 or start >= self.frame_count or end >= self.frame_count:
			raise ValueError(f"Start and end frames must be within the video length: {self.name}/{start}-{end}")

		# Check if start/end are in the correct order
		if start > end:
			raise ValueError(f"Start frame must be before the end frame: {self.name}/{start}-{end}")

		clip_name = f"{self.name}-{start}.mp4"
		clip_path = os.path.join(self.output_dir, clip_name)

		# Check if clip was already exported
		if os.path.exists(clip_path):
			log.debug(f"Clip {clip_name} already exists")
			return
		else:
			log.debug(f"Exporting {clip_name}")

		if sim:
			return

		# Set the start frame
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

		# Initialize the video writer
		out_frame_size = self.output_size or self.frame_size
		writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, out_frame_size)

		# Read and write the frames to the output video file
		for i in range(start, end + 1):

			ok, frame = self.cap.read()
			if not ok:
				break

			# Sharpen the frame
			frame = sharpen(frame)

			# Resize the frame
			if self.output_size is not None:
				frame = cv2.resize(frame, self.output_size)

			writer.write(frame)

		# Release the video writer
		writer.release()

		return

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
			start = row.ts_start
			end = row.ts_end

			self.export(start, end, sim=sim)

		return

	def clip_name(self, start: Union[int, str], end: Union[int, str]) -> str:
		"""
		Returns the name of a clip based on the start and end frame numbers.
		"""
		start, end = self.__ensure_timestamps__(start, end)
		return f"{self.name}-{start}-{end}.mp4"


def sharpen(image: np.ndarray) -> np.ndarray:
	# Apply the unsharp mask
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	return cv2.filter2D(image, -1, kernel)
