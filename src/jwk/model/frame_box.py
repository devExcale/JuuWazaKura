import cv2
import numpy as np

WHITE = np.array([255, 255, 255])
BLUE = np.array([0, 0, 255])


class FrameBox:

	def __init__(
			self,
			x1: int,
			y1: int,
			x2: int,
			y2: int,
	) -> None:
		"""
		Initializes the FrameBox object with the bounding box coordinates.
		"""

		self.x1: int = x1
		""" X coordinate of the top-left corner. """

		self.y1: int = y1
		""" Y coordinate of the top-left corner. """

		self.x2: int = x2
		""" X coordinate of the bottom-right corner. """

		self.y2: int = y2
		""" Y coordinate of the bottom-right corner. """

		self.w: int = x2 - x1
		""" Width of the bounding box. """

		self.h: int = y2 - y1
		""" Height of the bounding box. """

		self.area: int = self.w * self.h
		""" Area of the bounding box. """

		self.com_x: int = (x1 + x2) // 2
		""" X coordinate of the center of mass. """

		self.com_y: int = (y1 + y2) // 2
		""" Y coordinate of the center of mass. """

		return

	def intersects(self, o: 'FrameBox') -> bool:
		"""
		Checks if two bounding boxes intersect, based on a simple version of the Separating Axis Theorem.
		https://stackoverflow.com/a/40795835

		:param o: Other bounding box.
		:return: True if the bounding boxes intersect, False otherwise.
		"""

		no_intersect = self.x2 < o.x1 or self.x1 > o.x2 or self.y2 < o.y1 or self.y1 > o.y2

		return not no_intersect

	def athlete_score(
			self,
			frame: np.ndarray,
			thickness: int = 3,
			n_cols: int = 1,
			n_rows: int = 1,
	) -> float:
		"""
		Calculates the score of a bounding box based on its distance from the bottom and its area.
		Moreover, the score is scaled inversely by the distance from white and blue colors.
		The mean of three evenly-spaced columns and rows is used to calculate the distance from white and blue.

		:param frame: Frame to compute the score.
		:param thickness: Thickness of the columns and rows.
		:param n_cols: Number of columns to compute the mean.
		:param n_rows: Number of rows to compute the mean.
		:return: Score of the bounding box.
		"""

		h_frame, w_frame, _ = frame.shape

		# Calculate the area of the bounding box
		d_south = h_frame - self.y2

		# Compute how many rows and columns fit in the box
		n_cols = min(n_cols, self.w // thickness)
		n_rows = min(n_rows, self.h // thickness)

		col_spacing = (self.w - thickness * n_cols) // (n_cols + 1)
		row_spacing = (self.h - thickness * n_rows) // (n_rows + 1)

		avgs = []
		norms = []

		# Compute mean
		for i in range(n_cols):
			x_start = self.x1 + (i + 1) * col_spacing
			col = frame[self.y1:self.y2, x_start:x_start + thickness]
			avgs.append(np.mean(col, axis=(0, 1)))

		for i in range(n_rows):
			y_start = self.y1 + (i + 1) * row_spacing
			row = frame[y_start:y_start + thickness, self.x1:self.x2]
			avgs.append(np.sum(row, axis=(0, 1)))

		# Distances from white and blue
		for mean in avgs:
			norms.append(np.linalg.norm(mean - WHITE))
			norms.append(np.linalg.norm(mean - BLUE))

		# Calculate the score
		score = self.area / (d_south * d_south + 1)
		score *= 1 / (min(norms))

		return score

	def square(self) -> 'FrameBox':
		"""
		Converts the bounding box to a square by expanding the shorter side.

		:return: A new square bounding box.
		"""

		x1, y1 = self.x1, self.y1
		x2, y2 = self.x2, self.y2
		w, h = self.w, self.h

		if w > h:
			diff = w - h
			y1 -= diff // 2
			y2 += diff - diff // 2
		else:
			diff = h - w
			x1 -= diff // 2
			x2 += diff - diff // 2

		return FrameBox(x1, y1, x2, y2)

	def expand(self, pixels: int) -> 'FrameBox':
		"""
		Expands the bounding box by a number of pixels in each direction.

		:param pixels: Number of pixels to expand the bounding box.
		:return: Expanded bounding box.
		"""

		x1 = self.x1 - pixels
		y1 = self.y1 - pixels
		x2 = self.x2 + pixels
		y2 = self.y2 + pixels

		return FrameBox(x1, y1, x2, y2)

	def point_within(self, x: int, y: int) -> bool:
		"""
		Checks if a point is within the bounding box.

		:param x: X coordinate of the point.
		:param y: Y coordinate of the point.
		:return: True if the point is within the bounding box, False otherwise.
		"""

		return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

	def smooth(self, prev_frames: list['FrameBox']) -> 'FrameBox':
		"""
		Smooth the bounding box based on the previous frames.

		:param prev_frames: List of previous bounding boxes.
		:return: Smoothed bounding box.
		"""

		if not prev_frames or len(prev_frames) < 2:
			return self

		box_1 = prev_frames[-1]
		box_2 = prev_frames[-2]

		if not box_1.point_within(self.com_x, self.com_y) or not box_2.point_within(self.com_x, self.com_y):
			return box_1

		return self


def draw_box(
		frame: np.ndarray,
		x1: int,
		y1: int,
		x2: int,
		y2: int,
		text: str | None = None,
		color: tuple[int, int, int] = (0, 255, 0),
		thickness: int = 2,
) -> np.ndarray:
	"""
	Write a bounding box with text on the frame.

	:param frame: Frame to write on.
	:param x1: Top-left x-coordinate.
	:param y1: Top-left y-coordinate.
	:param x2: Bottom-right x-coordinate.
	:param y2: Bottom-right y-coordinate.
	:param text: Text to write.
	:param color: Color of the bounding box.
	:param thickness: Thickness of the bounding box.
	:return: Frame with bounding box and text.
	"""

	# Clip the coordinates
	x1 = max(0, x1)
	y1 = max(0, y1)
	x2 = min(frame.shape[1], x2)
	y2 = min(frame.shape[0], y2)

	# Draw the bounding box
	cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

	# Draw text
	if text:
		cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

	return frame
