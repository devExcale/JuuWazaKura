import json

import click

from .environment import MyEnv


@click.command(name='env')
def cmd_printenv() -> None:
	"""
	Prints the environment variables

	:return:
	"""

	items = {
		k: v
		for k, v in MyEnv.values().items()
	}

	print(json.dumps(items, indent=4))


switch = {
	'printenv': cmd_printenv
}

if __name__ == '__main__':
	cmd_printenv()
