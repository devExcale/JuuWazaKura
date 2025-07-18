import logging
import os

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..model import FrameBox, Tracker
from ..utils import MyEnv, get_logger, Drawing

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())

# Colors
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def video_annotate_objects(model: str, video_path: str, output_path: str):
	"""

	"""

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

	# Initialize the tracker
	tracker = Tracker(4, 4)

	while cap.isOpened():

		# Read the frame
		ret, frame_in = cap.read()
		if not ret:
			break

		frame_out = frame_in.copy()

		classes, scores, boxes, i_max_score, at_box = frame_inference(model, frame_in, tracker)

		params_draw_colors: list[tuple] = []

		# Annotate the frame
		for i, cls, box, score in zip(range(len(classes)), classes, boxes, scores):
			bb_color = GREEN if i == i_max_score else BLUE

			box: FrameBox
			# km_colors, _ = box.kmeans(frame_in, n_colors=8, n_iter=20)

			# params_draw_colors.append((frame_out, box.x1, box.y1, box.x2, box.y2, km_colors, 7))
			Drawing.box(frame_out, box.x1, box.y1, box.x2, box.y2, f'{cls} - {score:.4f}', bb_color)

		# Draw the bounding box containing the athletes
		Drawing.box(frame_out, at_box.x1, at_box.y1, at_box.x2, at_box.y2, None, RED, 2)

		# Draw the colors after the bounding boxes
		for params in params_draw_colors:
			Drawing.colors(*params)

		# Write the frame to the output video
		out.write(frame_out)

	# Release resources
	cap.release()
	out.release()

	log.info(f"Saved output video file.")

	return


def video_annotate_objects_simple(model: str, video_path: str, output_path: str):
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

	while cap.isOpened():

		# Read the frame
		ret, frame_in = cap.read()
		if not ret:
			break

		results = model.predict(source=frame_in, verbose=False)
		result = results[0]

		out.write(result.plot())

	cap.release()
	out.release()

	log.info(f"Saved output video file: {output_path}")

	return


def frame_inference(
		model: YOLO,
		frame: np.ndarray,
		tracker: Tracker | None = None,
) -> tuple[list[int], list[float], list[FrameBox], int, FrameBox]:
	"""
	Perform object detection on a single frame.

	:param model: YOLO model.
	:param frame: Frame to perform object detection on.
	:param tracker: Kalman filter tracker.
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

	# Keep only the lowest 5 boxes on the y axis
	boxes = sorted(boxes, key=lambda b: b.y2, reverse=True)[:6]

	# scores = [box.athlete_score(frame, thickness=10, n_rows=4, n_cols=4) for box in boxes]
	scores = [box.gi_likelihood(frame) for box in boxes]
	i_max_score = np.argmax(scores)

	# Get the bounding box containing the athletes
	box_pre: FrameBox = boxes_intersects(boxes, scores, i_max_score)
	box_post: FrameBox

	# Smooth the bounding box
	if tracker is not None:
		pred = tracker.predict(box_pre.as_np().reshape(-1, 1))
		box_post = FrameBox(*map(int, pred))
	else:
		box_post = box_pre

	# Other box processing
	box_post = box_post.square().expand(pixels=5)

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
	parent_segments: str = MyEnv.dataset_segments
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


def apply_filter(competition: str) -> None:
	# List competition segments
	parent_segments: str = MyEnv.dataset_segments
	segments: list[str] = os.listdir(os.path.join(parent_segments, competition))

	# Check output directory
	output_dir = os.path.join(parent_segments, f"{competition}-filter")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for filename in segments:
		# Get the input and output paths
		input_path = os.path.join(parent_segments, competition, filename)
		output_path = os.path.join(parent_segments, f"{competition}-filter", filename)

		# Annotate the video
		video_filter_huesat(input_path, output_path)

	return


def video_filter_huesat(video_path: str, output_path: str) -> None:
	# Open the video file
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		log.error(f"Error opening video file: {video_path}")
		return

	log.debug(f"Opened video file: {video_path}")

	hsv = 'hsv'

	# Get video properties
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'avc1')
	path_no_ext, ext = os.path.splitext(output_path)
	out_sv = cv2.VideoWriter(f"{path_no_ext}_sv{ext}", fourcc, fps, (width, height))
	out_hv = cv2.VideoWriter(f"{path_no_ext}_hv{ext}", fourcc, fps, (width, height))
	out_hs = cv2.VideoWriter(f"{path_no_ext}_hs{ext}", fourcc, fps, (width, height))

	log.debug(f"Output video file: {output_path}")

	while cap.isOpened():

		# Read the frame
		ret, frame_in = cap.read()
		if not ret:
			break

		frame_out = frame_in.copy()

		# Write the frame to the output video
		out_sv.write(hsv_flatten(frame_in, 0))
		out_hv.write(hsv_flatten(frame_in, 1))
		out_hs.write(hsv_flatten(frame_in, 2))

	# Release resources
	cap.release()
	out_sv.release()
	out_hv.release()
	out_hs.release()

	log.info(f"Saved output video file.")

	return


def hsv_flatten(frame_in: np.ndarray, idx: int) -> np.ndarray:
	"""
	Flatten the HSV channels and return the frame in BGR color space.

	:param frame_in: Input frame.
	:param idx: Index of the channel to flatten (0: hue, 1: saturation, 2: value).
	:return: Frame with the specified channel flattened.
	"""

	# Convert the frame to HSV color space
	hsv_frame = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)

	# Flatten the specified channel
	hsv_frame[:, :, idx] = 128

	# Convert back to BGR color space
	frame_out = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

	return frame_out
