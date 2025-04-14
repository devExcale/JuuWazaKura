import argparse
import os.path

from ..utils import MyEnv
from .yolov11 import annotate_competition_segments, apply_filter
from .gipick import main_picker, main_export, main_histogram

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument(
	'action',
	choices=['yolov11', 'filter', 'gi-export', 'gi-pick', 'gi-hist'],
	help='Action to perform',
)


def main_yolov11():
	annotate_competition_segments("yolo11s.pt", "LIG2202D1T3")

	return


def main_filter():
	apply_filter("yolo11s.pt", "TLIG2202D1T3")

	return


def main_gi_export():
	ds_folder = os.path.join(MyEnv.dataset_clips, "TLIG2202D1T3")
	gi_folder = os.path.join(MyEnv.dataset_source, "gi")
	main_export(ds_folder, gi_folder, "yolo11s.pt")

	return


def main_gi_pick():
	gi_folder = os.path.join(MyEnv.dataset_source, "gi")
	main_picker(gi_folder)

	return


def main_gi_hist():
	gi_folder = os.path.join(MyEnv.dataset_source, "gi")
	main_histogram(gi_folder, hue_bins=90, sat_bins=128)

	return


def main():
	switch = {
		'yolov11': main_yolov11,
		'filter': main_filter,
		'gi-export': main_gi_export,
		'gi-pick': main_gi_pick,
		'gi-hist': main_gi_hist,
	}

	args = parser.parse_args()
	main_method = switch.get(args.action, parser.print_help)

	main_method()

	return


if __name__ == '__main__':
	main()
