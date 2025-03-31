import argparse

from .yolov11 import annotate_competition_segments

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument(
	'action',
	choices=['yolov11'],
	help='Action to perform',
)


def main_yolov11():
	annotate_competition_segments("yolo11s.pt", "LIG2202D1T3")

	return


def main():
	switch = {
		'yolov11': main_yolov11,
	}

	args = parser.parse_args()
	main_method = switch.get(args.action, parser.print_help)

	main_method()

	return


if __name__ == '__main__':
	main()
