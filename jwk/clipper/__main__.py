import click

from .clipper import Clipper


@click.group(name="clipper")
def cmd_clipper():
	"""
	Clipper command line interface for generating video clips from timestamps.
	"""

	return


@cmd_clipper.command(name="extract")
@click.option(
	'--video', '-v',
	'video_path',
	type=str,
	required=True,
	help='Path to the video file'
)
@click.option(
	'--csv', '-t',
	'csv_path',
	type=str,
	required=True,
	help='Path to the CSV file containing the timestamps'
)
@click.option(
	'-o',
	'save_dir',
	type=str,
	default='',
	help='Output directory, defaults to the current directory'
)
@click.option(
	'--name', '-n',
	type=str,
	default=None,
	help='Prefix of the output clips, defaults to the filename of the video'
)
@click.option(
	'--width', '-w',
	type=int,
	default=None,
	help=(
			'Width of the output clips, defaults to the width of the input video. '
			'If the height isn\'t specified, the aspect ratio is preserved.'
	)
)
@click.option(
	'--height', '-g',
	type=int,
	default=None,
	help=(
			'Height of the output clips, defaults to the height of the input video. '
			'If the width isn\'t specified, the aspect ratio is preserved.'
	)
)
@click.option(
	'--default-ms-end', '-p',
	type=int,
	default=0,
	help='If set, adds 999 milliseconds to the end timestamp if no milliseconds are provided.',
)
def cmd_extract(
		video_path: str,
		csv_path: str,
		save_dir: str | None = None,
		name: str | None = None,
		width: str | None = None,
		height: str | None = None,
		default_ms_end: int = 0,
) -> None:
	"""
	Generate video clips from a video file based on timestamps provided in a CSV file.

	:param video_path: Path to the video file from which to clip segments.
	:param csv_path: Path to the CSV file containing the timestamps for clipping.
	:param save_dir: Directory where the output clips will be saved. Defaults to the current directory.
	:param name: Prefix for the output clips. If not specified, it defaults to the filename of the video without extension.
	:param width: Width of the output clips. If specified, it overrides the width of the input video.
	:param height: Height of the output clips. If specified, it overrides the height of the input video.
	:param default_ms_end: Default milliseconds to add to the end timestamp if no milliseconds are provided.
	:return: ``None``
	"""

	# Set default output name
	if name is None:
		name = '.'.join(video_path.split('/')[-1].split('.')[:-1])

	# Get resize dimensions
	if width is not None and height is not None:
		output_size = (width, height)
	elif width is not None:
		output_size = (width, None)
	elif height is not None:
		output_size = (None, height)
	else:
		output_size = None

	with Clipper(
			input_filepath=video_path,
			savedir=save_dir,
			name=name,
			size=output_size,
			default_ms_end=default_ms_end,
	) as clipper:
		clipper.export_from_csv(csv_path, sim=False)


if __name__ == '__main__':
	cmd_clipper()
