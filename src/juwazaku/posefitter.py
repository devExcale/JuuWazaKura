from typing import Optional

import cv2
from mediapipe import solutions as mp_solutions
from mediapipe.tasks.python.vision import PoseLandmarkerResult
from numpy import ndarray
from ultralytics import YOLO

from clipcommons import ClipCommons

mp_drawing = mp_solutions.drawing_utils
mp_pose = mp_solutions.pose

pose_landmarker_path = 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\pose_landmarker_heavy.task'


class PoseFitter(ClipCommons):

	def __init__(
			self,
			input_filepath: str,
			name: str,
			savedir: Optional[str] = None,
			print_fit: bool = False,
			export_print: bool = True,
	) -> None:

		self.input_filepath: str = input_filepath
		self.name = name
		self.savedir: Optional[str] = savedir
		self.print_fit: bool = print_fit
		self.export_print: bool = export_print

		self.yolo: Optional[YOLO] = None
		self.current_frame_bgr: Optional[ndarray] = None
		self.current_results: Optional[PoseLandmarkerResult] = None

	def __enter__(self) -> 'PoseFitter':

		# Open video file
		cap = cv2.VideoCapture(self.input_filepath)

		# Test if the video file is opened successfully
		if not cap.isOpened():
			raise FileNotFoundError(f"Could not open video file at path: {self.input_filepath}")

		self.cap = cap

		# Get video properties
		self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.fps = cap.get(cv2.CAP_PROP_FPS)

		self.yolo = YOLO('C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\yolov8x-pose.pt')

		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:
		"""
		Releases the video capture object.
		"""

		self.cap.release()

		return

	def do_next(self) -> bool:

		# Read frame from video
		ret, frame = self.cap.read()

		# No more frames
		if not ret:
			self.current_frame_bgr = None
			self.current_results = None
			return False

		# Assign frames
		self.current_frame_bgr = frame

		# Fit pose
		self.current_results = self.yolo(self.current_frame_bgr)

		return True

	def draw(self) -> ndarray:

		# Visualize the results on the frame
		annotated_frame = self.current_results[0].plot()

		return annotated_frame

	def export(self, debug: bool = False) -> None:
		"""
		Exports a clip from the opened video to another file.
		"""

		# Initialize the video writer
		out_filepath = f"{self.savedir}/{self.name}-posed.mp4"
		out = cv2.VideoWriter(out_filepath, cv2.VideoWriter_fourcc(*'MP4V'), self.fps, self.frame_size)

		# Read and write the frames to the output video file
		while True:

			if not self.do_next():
				break

			drawn_frame = self.draw()

			if debug:
				cv2.imshow('PoseFitter Debug', drawn_frame)
				if cv2.waitKey(10) & 0xFF == ord('q'):
					break

			out.write(self.draw())

		# Release the video writer
		out.release()

		if debug:
			cv2.destroyAllWindows()

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
