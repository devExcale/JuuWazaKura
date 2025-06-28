import logging
import unittest

from jwk.dataset import DatasetOrchestrator, DATASET_CUT_TRAINING, DATASET_CUT_TESTING
from jwk.utils import MyEnv, get_logger

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())


class TestDatasetOrchestrator(unittest.TestCase):

	def test_dataset_cuts(self) -> None:
		"""
		Test the dataset cuts for training and testing.

		:return: ``None``
		"""

		# Create dataset instances
		ds_all = DatasetOrchestrator()
		ds_train = DatasetOrchestrator(DATASET_CUT_TRAINING)
		ds_test = DatasetOrchestrator(DATASET_CUT_TESTING)

		# Load data
		ds_all.load_all(MyEnv.dataset_source, set(MyEnv.livefootage_include), set(MyEnv.livefootage_exclude))
		ds_train.load_all(MyEnv.dataset_source, set(MyEnv.livefootage_include), set(MyEnv.livefootage_exclude))
		ds_test.load_all(MyEnv.dataset_source, set(MyEnv.livefootage_include), set(MyEnv.livefootage_exclude))

		# Finalize datasets
		ds_all.finalize()
		ds_train.finalize()
		ds_test.finalize()

		# Numbers of segments
		expected_test = len(ds_all.set_testing)
		actual_all = len(ds_all.df)
		actual_train = len(ds_train.df)
		actual_test = len(ds_test.df)

		# Ensure number of test segments
		msg = f'Number of test segments does not match expected value: {actual_test} != {expected_test}'
		self.assertEqual(actual_test, expected_test, msg)

		# Ensure correct cut
		msg = f'Dataset train/test split does not match expected values: {actual_train} + {actual_test} != {actual_all}'
		self.assertEqual(actual_train + actual_test, actual_all, msg)

		return
