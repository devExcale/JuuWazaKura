import argparse

from .model import JwkModel

parser = argparse.ArgumentParser(description='Model')
parser.add_argument(
	'action',
	choices=['train'],
	help='Action to perform',
)


def main_train():

	# Example Usage
	handler = JwkModel()
	handler.fit_model(epochs=10, batch_size=4)
	handler.save_weights("model_weights.h5")

	return


def main():
	switch = {
		'train': main_train,
	}

	args = parser.parse_args()
	main_method = switch.get(args.action, parser.print_help)

	main_method()

	return


if __name__ == '__main__':
	main()
