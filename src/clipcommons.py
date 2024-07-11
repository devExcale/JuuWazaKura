from typing import Any


class ClipCommons:

	def __ensure_timestamps__(self, start: Any, end: Any) -> tuple[int, int]:

		if isinstance(start, str):
			start = int(ts_to_sec(start) * self.fps)

		if isinstance(end, str):
			end = int(ts_to_sec(end) * self.fps)

		if isinstance(start, int) and isinstance(end, int):
			return start, end

		raise ValueError("Start and end must be either integers or timestamps in the format '[HH:]MM:SS'")


def ts_to_sec(timestamp: str) -> int:
	"""
	Converts a timestamp string in the format 'HH:MM:SS' to seconds.
	Hours are optional.
	"""
	parts = timestamp.strip().split(':')

	# Check if the timestamp is in the format 'HH:MM:SS' or 'MM:SS'
	if len(parts) not in (2, 3):
		raise ValueError("Timestamp must be in the format 'HH:MM:SS' or 'MM:SS'")

	# Convert the timestamp parts to integers
	parts = [int(part) for part in parts]

	# Calculate the total seconds
	s = int(parts[-1])
	m = int(parts[-2])
	h = int(parts[-3]) if len(parts) == 3 else 0

	return h * 3600 + m * 60 + s
