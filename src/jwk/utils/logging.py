import logging

from datetime import datetime


class ApplicationLogFormatter(logging.Formatter):
	"""
	A custom formatter to mimic Spring Boot log style.
	"""

	def format(self, record: logging.LogRecord) -> str:
		timestamp = datetime.fromtimestamp(record.created).isoformat()

		level = record.levelname.upper().ljust(5)

		package_class = record.name.ljust(40)

		message = record.getMessage()

		return f"{timestamp} {level} [{package_class}] : {message}"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
	"""
	Configures a logger to mimic Spring Boot log style.
	"""

	# Get logger
	logger = logging.getLogger(name)
	logger.setLevel(level)

	# Add custom handler
	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.setFormatter(ApplicationLogFormatter())
		logger.addHandler(handler)

		# Do not propagate to the root logger
		logger.propagate = False

	return logger
