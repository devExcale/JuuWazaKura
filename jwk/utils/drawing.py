import cv2
import numpy as np


class Drawing:

	@staticmethod
	def box(
			frame: np.ndarray,
			x1: int,
			y1: int,
			x2: int,
			y2: int,
			text: str | None = None,
			color: tuple[int, int, int] = (0, 255, 0),
			thickness: int = 2,
	) -> None:
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

		return

	@staticmethod
	def colors(
			frame: np.ndarray,
			x1: int,
			y1: int,
			x2: int,
			y2: int,
			colors: set[tuple[int, int, int]],
			thickness: int = 2,
	) -> None:
		"""
		Draw the k-means colors on the bottom side of the bounding box.

		:param frame: Frame to write on.
		:param x1: Top-left x-coordinate.
		:param y1: Top-left y-coordinate.
		:param x2: Bottom-right x-coordinate.
		:param y2: Bottom-right y-coordinate.
		:param colors: Set of colors to draw.
		:param thickness: Side of a color square.
		:return: Frame with bounding box and colors.
		"""

		# Clip the coordinates
		x1 = max(0, x1)
		y2 = min(frame.shape[0], y2) + 1

		# Draw the colors
		for i, color in enumerate(colors):
			x = x1 + i * thickness
			color = tuple(int(i) for i in color)
			cv2.rectangle(frame, (x, y2), (x + thickness, y2 + thickness), color, -1)

		return
