from typing import Optional, Any, Tuple

import cv2
from numpy import ndarray
from ultralytics import YOLO
import pandas as pd

from ..commons import ClipCommons
from .arguments import PoseFitterArguments


class PoseFitter(ClipCommons):

	def __init__(
			self,
			args: PoseFitterArguments,
	) -> None:

		# Ensure the validity of the input arguments
		args.validity_barrier()

		self.args = args
		"Input arguments"

		self.yolo: Optional[YOLO] = None
		"YOLO model instance"

		self.cap: Optional[cv2.VideoCapture] = None
		"Input video capture object"

		self.current_frame_bgr: Optional[ndarray] = None
		"Last frame read, in BGR format"

		self.current_results: Optional[Any] = None
		"Last pose fitting results"

		self.fps: Optional[float] = None
		"Input (and output) video fps"

		self.frame_size: Optional[Tuple[int, int]] = None
		"Input (and output) video size (width, height)"

		self.frame_count: Optional[int] = None
		"Input video frame count"

	def __enter__(self) -> 'PoseFitter':

		print("Loading input video...")

		# Open video file
		cap = cv2.VideoCapture(self.args.input_video)

		# Test if the video file is opened successfully
		if not cap.isOpened():
			raise FileNotFoundError(f"Could not open video file at path: {self.args.input_video}")

		self.cap = cap

		# Get video properties
		self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.fps = cap.get(cv2.CAP_PROP_FPS)

		print("Loading YOLO model...")

		self.yolo = YOLO('res/yolov8s-seg.pt')

		print("Done.")

		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:
		"""
		Releases the video capture object.
		"""

		self.cap.release()

		return

	def fit_next(self) -> bool:
		"""
		Tries to read the next frame from the video and fit the pose.
		:return: Whether the frame was successfully read and fitted.
		"""

		# Read frame from video
		ret, frame = self.cap.read()

		# No more frames
		if not ret:
			self.current_frame_bgr = None
			self.current_results = None
			return False

		# Assign frame
		self.current_frame_bgr = frame

		# Fit pose
		self.current_results = self.yolo(self.current_frame_bgr)

		return True

	def export_preview_frame(self, vid_writer: cv2.VideoWriter) -> None:
		"""
		Exports the current frame results to the output preview video.
		:param vid_writer: Video writer object
		"""

		# Draw the pose on the frame
		annotated_frame = self.current_results[0].plot()

		# Write the frame into the output video
		vid_writer.write(annotated_frame)

	def export_results_frame(self, df: pd.DataFrame) -> None:
		"""
		Exports the current frame results to the output CSV file.
		:param df: DataFrame object
		"""

		pass

	def fit_and_export(self) -> None:
		"""
		Performs the export action(s) based on the input arguments.
		If export_preview is set, it will save a video with the pose fitting.
		If export_results is set, it will save a CSV file with the pose fitting results.
		"""

		# Variables initialization
		filepath_results: str = ''
		vid_writer: Optional[cv2.VideoWriter] = None
		df: Optional[pd.DataFrame] = None

		# Initialize preview export
		if self.args.export_preview:
			filepath_preview = f"{self.args.save_dir}/fitseg-{self.args.name}.mp4"
			vid_writer = cv2.VideoWriter(filepath_preview, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)

		# Initialize results export
		if self.args.export_results:
			filepath_results = f"{self.args.save_dir}/fitseg-{self.args.name}.csv"
			df = pd.DataFrame(columns=['frame', 'blue', 'white'])
			df.to_csv(filepath_results, index=False)  # Verify the file can be written

		print(f"Processing {self.args.name}")

		# Read and write the frames to the output video file
		while self.fit_next():

			if self.args.export_preview:
				self.export_preview_frame(vid_writer)

			if self.args.export_results:
				self.export_results_frame(df)

		# Release the video writer
		if self.args.export_preview:
			vid_writer.release()

		# Release the CSV file
		if self.args.export_results:
			df.to_csv(filepath_results, index=False)

		return

	def main_yolo_v8(self):

		while self.cap.isOpened():

			ret, frame = self.cap.read()
			if not ret:
				break

			# Media pose prediction
			results = self.yolo(frame)

			# Visualize the results on the frame
			annotated_frame = results[0].plot()

			# Write the frame into the output video
			# out.write(image)

			# Code to quit the video incase you are using the webcam
			cv2.imshow('Activity recognition', annotated_frame)
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		cv2.destroyAllWindows()
