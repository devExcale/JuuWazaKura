from .arguments import PoseFitterArguments
from .posefitter import PoseFitter


def main():
	# Get input arguments
	args = PoseFitterArguments().from_parser()

	# Print input arguments
	print(args)

	# Run the pose fitting
	with PoseFitter(args=args) as posefitter:
		posefitter.fit_and_export()


if __name__ == '__main__':
	main()
