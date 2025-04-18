import logging
import unittest

from jwk.dataset import DatasetHandler, DatasetBatchGenerator
from jwk.utils import MyEnv, get_logger

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())


class TestDatasetBatchGenerator(unittest.TestCase):

	def setUp(self):
		self.dataset = DatasetHandler()
		self.dataset.load_all(MyEnv.dataset_source, set(MyEnv.dataset_include), set(MyEnv.dataset_exclude))
		self.dataset.finalize()

		# Initialize DatasetBatchGenerator
		self.batch_generator = DatasetBatchGenerator(
			dataset=self.dataset,
			segments_per_batch=1,
			input_size=(112, 112),
			window_frames=8,
			workers=4,
			use_multiprocessing=True,
		)

	def test_batch_shape(self):
		log.info("test_batch_shape: Start")

		for i in range(len(self.batch_generator)):
			# Get batch
			x_batch, y_batch = self.batch_generator[i]

			log.debug(f"Batch {i}: x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")

			batch_size = x_batch.shape[0]
			n_frames = x_batch.shape[1]
			frame_shape = x_batch.shape[2:]

			# Assert sizes
			self.assertEqual(batch_size, 1)
			self.assertTrue(n_frames > 0)
			self.assertTupleEqual(frame_shape, (112, 112, 3))

			log.debug(f'Batch {i}: passed')

		log.info("test_batch_shape: End")

		return
