import argparse

from .posefitter import PoseFitter

parser = argparse.ArgumentParser(
	description='Segment a video file in multiple clips based on the timestamps in a dedicated CSV file.'
)

parser.add_argument(
	'--video', '-v',
	dest='video_path',
	type=str,
	help='Path to the video file',
	required=True,
)
parser.add_argument(
	'-o',
	dest='save_dir',
	type=str,
	help='Output directory, defaults to the current directory',
	required=False,
)
parser.add_argument(
	'--name', '-n',
	dest='name',
	type=str,
	help='Prefix of the output clips, defaults to the filename of the video',
	required=False,
)


def main():
	# Get input arguments
	args = parser.parse_args()

	# Set default output directory
	if args.save_dir is None:
		args.save_dir = ''

	# Set default output name
	if args.name is None:
		args.name = '.'.join(args.video_path.split('/')[-1].split('.')[:-1])

	# Print input arguments
	print(args)

	with PoseFitter(args.video_path, args.name, args.save_dir) as posefitter:
		posefitter.export(debug=False)


if __name__ == '__main__':
	main()
