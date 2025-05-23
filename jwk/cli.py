import click

from .dataset.__main__ import cmd_dataset
from .devtools.__main__ import cmd_devtools
from .model.__main__ import cmd_model
from .utils.__main__ import cmd_printenv


@click.group()
def cli() -> None:
	"""
	CLI command line interface.
	"""

	return


# List of subcommands
subcommands = [
	cmd_dataset,
	cmd_devtools,
	cmd_model,
	cmd_printenv,
]

# Register subcommands
for cmd in subcommands:
	# noinspection PyTypeChecker
	cli.add_command(cmd)
