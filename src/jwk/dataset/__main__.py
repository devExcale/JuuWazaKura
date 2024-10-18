import json

from .datasethandler import DatasetHandler


def main():
	# Create Handler
	handler = DatasetHandler()
	handler.config()

	# Load all the data
	handler.load_all()

	# Clean the data
	handler.clean_data()

	# Find unknown throws
	unknown_throws = handler.get_unknown_throws()

	# Compute statistics
	stats = handler.compute_stats()

	# Print statistics (beautify)
	print(json.dumps(stats, indent=4))

	# Print unknown throws
	print('Unknown throws:', unknown_throws)


if __name__ == '__main__':
	main()
