def extract_filename(file_path: str) -> str:
	return '.'.join(
		file_path.split('/')[-1]
		.split('.')[:-1]
	)
