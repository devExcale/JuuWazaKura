import json
import logging
import os
from typing import Dict

from yt_dlp import YoutubeDL

log = logging.getLogger(__name__)


def download_dataset(dir_dataset: str):
	# Ensure dataset directory exists
	if not os.path.exists(dir_dataset):
		log.error(f'Dataset directory does not exist: {dir_dataset}')
		return

	log.debug(f'Dataset directory: {dir_dataset}')

	# Read dataset json file
	with open(os.path.join(dir_dataset, 'dataset.json')) as f:
		ds = json.load(f)

	# Get dataset parameters
	file_format: str = str(ds['fileFormat'])
	vid_ids: Dict[str, str] = ds['vidIds']
	n_vids = len(vid_ids)

	# Create the directory for the videos
	dir_vids = os.path.join(dir_dataset, 'vids')
	if not os.path.exists(dir_vids):
		os.makedirs(dir_vids)

	log.debug(f'Downloading dataset to "{dir_vids}"')

	n_downloaded = 0

	# Download all the files
	for i, (vid_key, vid_id) in enumerate(vid_ids.items()):
		# Check whether a corresponding csv exists
		vid_csv = os.path.join(dir_dataset, f'{vid_key}.csv')
		if not os.path.exists(vid_csv):
			log.warning(f'Skipping {vid_key} as the corresponding clippings do not exist')
			continue

		# Create the yt-dlp options
		ydl_opts = {
			'outtmpl': os.path.join(dir_dataset, 'vids', f'{vid_key}.%(ext)s'),
			'format': file_format,
			'quiet': True,
			'concurrent_fragment_downloads': 8,
		}

		log.info(f'Downloading {padded_ratio(i + 1, n_vids)}: {vid_id}')

		# Download the file
		with YoutubeDL(ydl_opts) as ydl:
			ydl.download([vid_id])

		n_downloaded += 1

	log.info(f'Downloaded {n_downloaded}/{n_vids} videos')


def padded_ratio(n: int, top: int, pad: str = " ") -> str:
	digits = lambda x: len(str(x))
	return pad * (digits(top) - digits(n)) + str(n) + '/' + str(top)
