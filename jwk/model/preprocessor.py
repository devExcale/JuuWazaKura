import os

import cv2
import numpy as np
from ultralytics import YOLO

from .frame_box import FrameBox
from .tracker import Tracker
from ..utils import MyEnv


class JwkPreprocessor:
	"""
	Video preprocessor for JwkModel.
	"""

	def __init__(self):
		"""
		Initialize the preprocessor object.
		"""

		# Check model weight file
		if not MyEnv.yolo_model.endswith('.pt') or not os.path.isfile(MyEnv.yolo_model):
			raise ValueError(f"Invalid YOLO model path: {MyEnv.yolo_model}")

		self.model = YOLO(MyEnv.yolo_model, verbose=False)
		""" YOLO model object for people detection. """

		pass

	def preprocess_video(
			self,
			input_path: str,
			output_path: str,
	) -> None:
		"""
		Preprocess the specified input video and save the output to the specified path.

		:param input_path: Path to the input video.
		:param output_path: Path to save the output video.
		:return: ``None``
		"""

		# Open input video file
		cap = cv2.VideoCapture(input_path)
		if not cap.isOpened():
			raise ValueError(f"Error opening video file: {input_path}")

		# Get video properties
		fps = cap.get(cv2.CAP_PROP_FPS)

		# Open output video file
		fourcc = cv2.VideoWriter_fourcc(*'avc1')
		out = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))

		# Initialize the tracker
		tracker = Tracker(4, 4)

		# Process each frame
		while cap.isOpened():

			# Read the frame
			ret, frame_in = cap.read()
			if not ret:
				break

			# Preprocess the frame
			frame_out = self.preprocess_frame(frame_in, tracker)

			# Resize the frame to 224x224
			frame_out = cv2.resize(frame_out, (224, 224))

			# Write the output frame
			out.write(frame_out)

		# Release resources
		cap.release()
		out.release()

		return

	def preprocess_frame(
			self,
			frame: np.ndarray,
			tracker: Tracker | None,
			debug: dict | None = None
	) -> np.ndarray:
		"""
		Looks for the athletes in the frame and returns the area as a squared frame.

		:param frame: Input frame.
		:param tracker: Kalman filter tracker.
		:param debug: Dictionary where to dump debug information (if needed).
		:return: Squared subframe with the athlete (hopefully).
		"""

		# Perform object detection
		results = self.model.predict(frame, verbose=False)

		classes: list[int] = []
		""" List of detected classes. """
		boxes: list[FrameBox] = []
		""" List of bounding boxes """

		# Parse results
		for result in results:
			for box in result.boxes:
				classes.append(int(box.cls.item()))
				xyxy = box.xyxy
				x1 = int(xyxy[0, 0])
				y1 = int(xyxy[0, 1])
				x2 = int(xyxy[0, 2])
				y2 = int(xyxy[0, 3])
				boxes.append(FrameBox(x1, y1, x2, y2))

		filter_n_ymax = MyEnv.preprocess_n_ymax if MyEnv.preprocess_n_ymax > 0 else len(boxes)

		# Sort by y2 and keep only the lowest n ones
		idxs = np.argsort([box.y2 for box in boxes])[::-1][:filter_n_ymax]
		boxes = [boxes[i] for i in idxs]
		classes = [classes[i] for i in idxs]

		# Compute scores
		# noinspection PyTypeChecker
		scores = [box.athlete_score_v3(frame) for box in boxes]
		idx_main_box = np.argmax(scores)

		# Get the bounding box containing the athletes
		box_pre: FrameBox = intersect_boxes(idx_main_box, boxes, scores)
		box_post: FrameBox

		# Apply tracking to the area
		if tracker is not None:
			pred = tracker.predict(box_pre.as_np().reshape(-1, 1))
			box_post = FrameBox(*map(int, pred))
		else:
			box_post = box_pre

		# Further processing
		box_post = box_post.square().expand(pixels=10)

		# Add debug info if requested
		if debug is not None:
			debug['classes'] = classes
			debug['boxes'] = boxes
			debug['scores'] = scores
			debug['idx_main_box'] = idx_main_box
			debug['main_box'] = box_pre
			debug['final_box'] = box_post

		return box_post.slice(frame)


def intersect_boxes(
		idx_main_box: int,
		boxes: list[FrameBox],
		scores: list[float],
		threshold: float = 0.3
) -> 'FrameBox':
	"""
	Given a main box, compute the area containing the boxes that directly intersect with the main box
	and have a score at least above a certain threshold of the main box.

	:param idx_main_box: Index of the main box.
	:param boxes: List of boxes.
	:param scores: List of scores.
	:param threshold: Threshold percentage to consider a box as intersecting.
	:return: Bounding box containing all the boxes that intersect with the main box.
	"""

	main_box = boxes[idx_main_box]
	x_min = main_box.x1
	y_min = main_box.y1
	x_max = main_box.x2
	y_max = main_box.y2

	min_score = threshold * scores[idx_main_box]

	for j, box in enumerate(boxes):

		if idx_main_box == j:
			continue

		# Include only boxes with a score at least 30% of the i-th box
		if scores[j] < min_score:
			continue

		# Check if the boxes intersect
		if main_box.intersects(box):
			x_min = min(x_min, box.x1)
			y_min = min(y_min, box.y1)
			x_max = max(x_max, box.x2)
			y_max = max(y_max, box.y2)

	return FrameBox(x_min, y_min, x_max, y_max)
