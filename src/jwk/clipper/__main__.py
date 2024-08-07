import argparse

from .clipper import Clipper

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
	'--csv', '-t',
	dest='csv_path',
	type=str,
	help='Path to the CSV file containing the timestamps',
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
parser.add_argument(
	'--width', '-w',
	dest='width',
	type=int,
	help='Width of the output clips, defaults to the width of the input video. '
		 'If the height isn\'t specified, the aspect ratio is preserved.',
	required=False,
)
parser.add_argument(
	'--height', '-g',
	dest='height',
	type=int,
	help='Height of the output clips, defaults to the height of the input video. '
		 'If the width isn\'t specified, the aspect ratio is preserved.',
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

	# Get resize dimensions
	if args.width is not None and args.height is not None:
		args.output_size = (args.width, args.height)
	elif args.width is not None:
		args.output_size = (args.width, None)
	elif args.height is not None:
		args.output_size = (None, args.height)
	else:
		args.output_size = None

	with Clipper(args.video_path, args.save_dir, args.name, args.output_size) as clipper:
		clipper.export_from_csv(args.csv_path, sim=False)


if __name__ == '__main__':
	main()
