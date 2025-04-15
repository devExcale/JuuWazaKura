import cv2
import numpy as np


class Tracker:
	"""
	This class represents a tracker object that uses OpenCV and Kalman Filters.
	"""

	def __init__(self, n_states: int, n_measures: int) -> None:
		"""
		Initializes the Tracker object.

		:param n_states: Number of state variables.
		:param n_measures: Number of measurement variables.
		"""

		self.n_states = n_states
		""" Number of state variables. """
		self.n_measures = n_measures
		""" Number of measurement variables. """

		self.first_predict: bool = True
		""" Whether this is the first run of the tracker. """

		# Create a Kalman filter object with specified parameters
		self.kf = cv2.KalmanFilter(n_states, n_measures)
		""" Kalman filter object. """

		# Set the measurement matrix of the Kalman filter.
		self.kf.measurementMatrix = np.eye(self.n_measures, self.n_measures, dtype=np.float32)

		# Set the transition matrix of the Kalman filter.
		self.kf.transitionMatrix = np.identity(self.n_states, dtype=np.float32)

		# Set the process noise covariance matrix of the Kalman filter.
		self.kf.processNoiseCov = np.identity(self.n_states, dtype=np.float32) * 0.03

		return

	def predict(self, measurement: np.ndarray | None) -> np.ndarray | None:
		"""
		Corrects the Kalman filter with the given measurements.

		:param measurement: The next measurement to correct the Kalman filter.
		:return: The corrected state of the Kalman filter.
		"""

		# Initialize first filter measurement
		if self.first_predict:

			if measurement is None:
				return None

			self.first_predict = False
			self.kf.statePost = measurement
			return measurement

		# Get next state prediction
		prediction = self.kf.predict()

		# Correct the Kalman filter with the new measurement
		if measurement is not None:
			self.kf.correct(measurement)

		return prediction
