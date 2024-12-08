import argparse

parser = argparse.ArgumentParser(description='Commons')
parser.add_argument(
	'action',
	choices=['env'],
	help='Action to perform: env'
)


def main_env():
	from .my_env import MyEnv
	import json

	items = {
		k: v
		for k, v in MyEnv.values().items()
	}

	print(json.dumps(items, indent=4))


switch = {
	'env': main_env
}


def main():
	args = parser.parse_args()

	if args.action in switch:
		switch[args.action]()
	else:
		parser.print_help()


if __name__ == '__main__':
	main()
