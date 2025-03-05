def filename_from_path(filepath: str) -> str:
	return '.'.join(
		filepath.split('/')[-1]
		.split('.')[:-1]
	)
