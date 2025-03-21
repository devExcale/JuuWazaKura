import argparse
import logging

from .model import JwkModel
from ..utils import get_logger, MyEnv

# Initialize logging
log: logging.Logger = get_logger(__name__, MyEnv.log_level())

parser = argparse.ArgumentParser(description='Model')
parser.add_argument(
	'action',
	choices=['train'],
	help='Action to perform',
)


def main_train():
	# Example Usage
	handler = JwkModel()
	handler.fit_model(epochs=10)
	handler.save_weights("model.weights.h5")

	return


def main():
	switch = {
		'train': main_train,
	}

	log.info(f"Debug level: {logging.getLevelName(log.level)}")

	args = parser.parse_args()
	main_method = switch.get(args.action, parser.print_help)

	main_method()

	return


if __name__ == '__main__':
	main()
