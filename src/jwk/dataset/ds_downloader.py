import json
import logging
import os
from typing import Dict

from yt_dlp import YoutubeDL

from ..clipper import Clipper

log = logging.getLogger(__name__)


def download_dataset(dir_dataset: str) -> None:

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
	dir_vids = os.path.join(dir_dataset, 'clips')
	if not os.path.exists(dir_vids):
		os.makedirs(dir_vids)

	log.debug(f'Downloading dataset to "{dir_vids}"')

	downloaded = []

	# Download all the files
	for i, (vid_key, vid_id) in enumerate(vid_ids.items()):

		log.info(f'Downloading {padded_ratio(i + 1, n_vids)}: {vid_id}')

		if download_vid(dir_dataset, vid_key, vid_id, file_format):
			downloaded.append(vid_key)

	n_downloaded = len(downloaded)

	log.info(f'Downloaded {n_downloaded}/{n_vids} videos')

	# Convert full videos to clips
	for i, vid_key in enumerate(downloaded):

		log.info(f'Clipping {padded_ratio(i + 1, n_downloaded)}: {vid_key}')

		clipper = Clipper(
			input_filepath=os.path.join(dir_vids, f'{vid_key}.mp4'),
			savedir=dir_vids,
			name=vid_key,
		)

		with clipper as c:
			c.export_from_csv(os.path.join(dir_dataset, f'{vid_key}.csv'))


def download_vid(dir_dataset: str, vid_key: str, vid_id: str, file_format: str) -> bool:
	"""
	Download a video from YouTube given its ID.

	:param dir_dataset: Directory where the dataset is stored
	:param vid_key: Key of the video (filename)
	:param vid_id: ID of the video
	:param file_format: YT video format (`yt-dlp -F <ID>` for available formats)
	:return: True if the video was downloaded, False otherwise
	"""

	# Check whether a corresponding csv exists
	vid_csv = os.path.join(dir_dataset, f'{vid_key}.csv')
	if not os.path.exists(vid_csv):
		log.warning(f'Skipping {vid_key} as the corresponding clippings do not exist')
		return False

	# Check whether the video already exists
	vid_path = os.path.join(dir_dataset, 'clips', f'{vid_key}.mp4')
	if os.path.exists(vid_path):
		log.info(f'Skipping {vid_key} as the video already exists')
		return True

	# Create the yt-dlp options
	ydl_opts = {
		'outtmpl': os.path.join(dir_dataset, 'clips', f'{vid_key}.%(ext)s'),
		'format': file_format,
		'quiet': True,
		'concurrent_fragment_downloads': 8,
	}

	# Download the file
	with YoutubeDL(ydl_opts) as ydl:
		ret = ydl.download([vid_id])

	return ret == 0


def padded_ratio(n: int, top: int, pad: str = " ") -> str:
	digits = lambda x: len(str(x))
	return pad * (digits(top) - digits(n)) + str(n) + '/' + str(top)
