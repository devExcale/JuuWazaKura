import argparse
import json

from .environment import MyEnv

parser = argparse.ArgumentParser(description='Commons')
parser.add_argument(
	'action',
	choices=['printenv'],
	help='Action to perform: printenv'
)


def main_printenv() -> None:
	"""
	Prints the environment variables

	:return:
	"""

	items = {
		k: str(v)
		for k, v in MyEnv.values().items()
	}

	print(json.dumps(items, indent=4))


switch = {
	'printenv': main_printenv
}


def main() -> None:
	args = parser.parse_args()

	if args.action in switch:
		switch[args.action]()
	else:
		parser.print_help()

	return


if __name__ == '__main__':
	main()
