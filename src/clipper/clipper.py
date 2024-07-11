from typing import Union

import cv2
import pandas as pd

from clipcommons import ClipCommons


class Clipper(ClipCommons):
	"""
	Class for generating clips from a video file.
	"""

	def __init__(self, input_filepath: str, savedir: str, name: str, fps: float) -> None:
		"""
		Initializes the Clipper object with the video file path.
		"""
		self.input_filepath = input_filepath
		self.savedir = savedir
		self.name = name

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
			raise ValueError("Start and end frames must be within the video length")

		# Check if start/end are in the correct order
		if start > end:
			raise ValueError("Start frame must be before the end frame")

		if sim:
			return

		# Set the start frame
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

		# Initialize the video writer
		out_filepath = f"{self.savedir}/{self.name}-{start}-{end}.mp4"
		out = cv2.VideoWriter(out_filepath, cv2.VideoWriter_fourcc(*'MP4V'), self.fps, self.frame_size)

		# Read and write the frames to the output video file
		for i in range(start, end + 1):
			ret, frame = self.cap.read()
			if not ret:
				break

			out.write(frame)

		# Release the video writer
		out.release()

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
		df = pd.read_csv(csv_path, sep=' ', header=0)

		# Export clips based on the timestamps
		for row in df.itertuples():
			start = row.ts_start
			end = row.ts_end

			print(f"Exporting {start}-{end} to {self.clip_name(start, end)}")

			self.export(start, end, sim=sim)

		return

	def clip_name(self, start: Union[int, str], end: Union[int, str]) -> str:
		"""
		Returns the name of a clip based on the start and end frame numbers.
		"""
		start, end = self.__ensure_timestamps__(start, end)
		return f"{self.name}-{start}-{end}.mp4"
