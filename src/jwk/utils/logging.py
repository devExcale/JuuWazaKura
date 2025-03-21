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


def addLoggingLevel(levelName, levelNum, methodName=None):
	"""
	Comprehensively adds a new logging level to the `logging` module and the
	currently configured logging class.

	`levelName` becomes an attribute of the `logging` module with the value
	`levelNum`. `methodName` becomes a convenience method for both `logging`
	itself and the class returned by `logging.getLoggerClass()` (usually just
	`logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
	used.

	To avoid accidental clobberings of existing attributes, this method will
	raise an `AttributeError` if the level name is already an attribute of the
	`logging` module or if the method name is already present.

	https://stackoverflow.com/a/35804945

	Example
	-------
	>>> addLoggingLevel('TRACE', logging.DEBUG - 5)
	>>> logging.getLogger(__name__).setLevel("TRACE")
	>>> logging.getLogger(__name__).trace('that worked')
	>>> logging.trace('so did this')
	>>> logging.TRACE
	5

	"""
	if not methodName:
		methodName = levelName.lower()

	if hasattr(logging, levelName):
		raise AttributeError('{} already defined in logging module'.format(levelName))
	if hasattr(logging, methodName):
		raise AttributeError('{} already defined in logging module'.format(methodName))
	if hasattr(logging.getLoggerClass(), methodName):
		raise AttributeError('{} already defined in logger class'.format(methodName))

	# This method was inspired by the answers to Stack Overflow post
	# http://stackoverflow.com/q/2183233/2988730, especially
	# http://stackoverflow.com/a/13638084/2988730
	def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(levelNum):
			self._log(levelNum, message, args, **kwargs)

	def logToRoot(message, *args, **kwargs):
		logging.log(levelNum, message, *args, **kwargs)

	logging.addLevelName(levelNum, levelName)
	setattr(logging, levelName, levelNum)
	setattr(logging.getLoggerClass(), methodName, logForLevel)
	setattr(logging, methodName, logToRoot)


addLoggingLevel('TRACE', logging.DEBUG - 5)
