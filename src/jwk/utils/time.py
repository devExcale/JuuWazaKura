from re import match

regex_ts = r'^(?:\d+:)?\d{1,2}:\d{1,2}(?:\.\d{1,3})?$'


def get_framestamp(start: str | float, end: str | float, fps: float) -> tuple[int, int]:
	"""
	Converts start and end times to frame numbers.
	The times can either be seconds (up to 3 decimal places) or timestamps in the format '[HH:]MM:SS[.000]'.

	:param start: Start time
	:param end: End time
	:param fps: Frames per second
	:return: Tuple of start and end frame numbers
	"""

	if isinstance(start, str):
		start = int(ts_to_sec(start) * fps)

	if isinstance(end, str):
		end = int(ts_to_sec(end) * fps)

	if isinstance(start, int) and isinstance(end, int):
		return start, end

	raise ValueError("Start and end must be either integers or timestamps in the format '[HH:]MM:SS'")


def ts_to_sec(timestamp: str) -> float:
	"""
	Converts a timestamp string in the format '[HH:]MM:SS[.000]' to seconds.
	"""

	# Check format of the timestamp
	if not match(regex_ts, timestamp):
		raise ValueError("Timestamp must be format '[HH:]MM:SS[.000]'")

	parts = timestamp.strip().split(':')

	# Get hours if present
	h = int(parts[0]) if len(parts) == 3 else 0

	# Get minutes
	m = int(parts[-2])

	# Get seconds, and milliseconds if present
	s, ms = map(int, parts[-1].split('.')) if '.' in parts[-1] else (int(parts[-1]), 0)

	return h * 3600 + m * 60 + s + ms / 1000
