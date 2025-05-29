import tensorflow as tf

from keras.callbacks import Callback


class MemoryCleanupCallback(Callback):

	def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
		tf.keras.backend.clear_session()
		return
