import os

import cv2
import numpy as np

from ultralytics import YOLO
import logging

from ultralytics.engine.results import Results

from ..utils import MyEnv, get_logger
from ..model import FrameBox, draw_box

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())

# Colors
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def video_annotate_objects(model: str, video_path: str, output_path: str):
	# Load the YOLO model
	model = YOLO(model, verbose=False)

	# Open the video file
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		log.error(f"Error opening video file: {video_path}")
		return

	log.debug(f"Opened video file: {video_path}")

	# Get video properties
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'avc1')
	out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

	log.debug(f"Output video file: {output_path}")

	t_boxes = []
	""" List of inferred boxes. """

	while cap.isOpened():

		# Read the frame
		ret, frame = cap.read()
		if not ret:
			break

		classes, scores, boxes, i_max_score, at_box = frame_inference(model, frame, t_boxes)

		# Annotate the frame
		for i, cls, box, score in zip(range(len(classes)), classes, boxes, scores):
			color = GREEN if i == i_max_score else BLUE
			frame = draw_box(frame, box.x1, box.y1, box.x2, box.y2, f'{cls} - {score:.4f}', color)

		# Draw the bounding box containing the athletes
		frame = draw_box(frame, at_box.x1, at_box.y1, at_box.x2, at_box.y2, None, RED, 2)

		# Write the frame to the output video
		out.write(frame)

	# Release resources
	cap.release()
	out.release()

	log.info(f"Saved output video file.")

	return


def frame_inference(
		model: YOLO,
		frame: np.ndarray,
		inferred_boxes: list[FrameBox] | None = None,
) -> tuple[list[int], list[float], list[FrameBox], int, FrameBox]:
	"""
	Perform object detection on a single frame.

	:param model: YOLO model.
	:param frame: Frame to perform object detection on.
	:param inferred_boxes: List of inferred boxes.
	:return: Detected classes, athlete scores, bounding boxes, inferred index, inferred athlete box.
	"""

	# Perform object detection
	results: list[Results] = model(frame)

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

	scores = [box.athlete_score(frame, thickness=5, n_rows=2, n_cols=2) for box in boxes]
	i_max_score = np.argmax(scores)

	# Get the bounding box containing the athletes
	box_pre = boxes_intersects(boxes, scores, i_max_score)
	box_post = box_pre

	# Smooth the bounding box
	if inferred_boxes:
		box_post = box_post.smooth(inferred_boxes)

	# Other box processing
	box_post = box_post.square().expand(pixels=5)

	inferred_boxes.append(box_pre)

	return classes, scores, boxes, i_max_score, box_post


def boxes_intersects(boxes: list[FrameBox], scores: list[float], i: int) -> 'FrameBox':
	"""
	Returns the bounding box containing the boxes that intersect with the i-th box.
	"""

	i_box = boxes[i]
	x_min = i_box.x1
	y_min = i_box.y1
	x_max = i_box.x2
	y_max = i_box.y2

	threshold = 0.3 * scores[i]

	for j, box in enumerate(boxes):
		if i == j:
			continue

		# Include only boxes with a score at least 30% of the i-th box
		if scores[j] < threshold:
			continue

		# Check if the boxes intersect
		if i_box.intersects(box):
			x_min = min(x_min, box.x1)
			y_min = min(y_min, box.y1)
			x_max = max(x_max, box.x2)
			y_max = max(y_max, box.y2)

	return FrameBox(x_min, y_min, x_max, y_max)


def annotate_competition_segments(model: str, competition: str) -> None:
	# List competition segments
	parent_segments: str = MyEnv.dataset_clips
	segments: list[str] = os.listdir(os.path.join(parent_segments, competition))

	# Check output directory
	output_dir = os.path.join(parent_segments, f"{model}-{competition}")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for filename in segments:
		# Get the input and output paths
		input_path = os.path.join(parent_segments, competition, filename)
		output_path = os.path.join(parent_segments, f"{model}-{competition}", filename)

		# Annotate the video
		video_annotate_objects(model, input_path, output_path)

	return
