import os

import cv2
import numpy as np
from ultralytics import YOLO

from ..utils import MyEnv


def main_picker(gi_folder: str):
	"""
	Loop through all pictures in the folder. If the picture is already in a subfolder
	(white, blue, or bin), skip it. Otherwise, show the picture to the user with cv2
	and allow classification: b for blue, w for white, d for delete, esc for exit.
	"""

	subfolders = ['white', 'blue', 'bin']
	for subfolder in subfolders:
		os.makedirs(os.path.join(gi_folder, subfolder), exist_ok=True)

	for filename in os.listdir(gi_folder):
		file_path = os.path.join(gi_folder, filename)
		if not os.path.isfile(file_path):
			continue

		# Delete files already in subfolders
		if any(os.path.exists(os.path.join(gi_folder, subfolder, filename)) for subfolder in subfolders):
			os.remove(file_path)
			continue

		# Display the image
		img = cv2.imread(file_path)
		if img is None:
			continue

		# Resize and pad the image to fit in a 640x640 square
		h, w = img.shape[:2]
		scale = min(640 / w, 640 / h)
		img = cv2.resize(img, (int(w * scale), int(h * scale)))

		# Create a black square canvas
		img = cv2.copyMakeBorder(
			img,
			top=(640 - img.shape[0]) // 2,
			bottom=(640 - img.shape[0] + 1) // 2,
			left=(640 - img.shape[1]) // 2,
			right=(640 - img.shape[1] + 1) // 2,
			borderType=cv2.BORDER_CONSTANT,
			value=(0, 0, 0)
		)

		cv2.imshow('Image Picker', img)
		key = cv2.waitKey(0)

		if key == ord('b'):  # Move to blue folder
			os.rename(file_path, os.path.join(gi_folder, 'blue', filename))
		elif key == ord('w'):  # Move to white folder
			os.rename(file_path, os.path.join(gi_folder, 'white', filename))
		elif key == ord('d'):  # Move to bin folder
			os.rename(file_path, os.path.join(gi_folder, 'bin', filename))
		elif key == 27:  # ESC key to exit
			break

	cv2.destroyAllWindows()

	return


def main_export(ds_folder: str, gi_folder: str, model_path: str):
	"""
	Perform inference on all video files in the provided dataset folder.
	Export the top 5 bounding boxes (based on scores) to the gi folder.
	"""

	from .yolov11 import frame_inference

	os.makedirs(gi_folder, exist_ok=True)

	# Load the YOLO model
	model = YOLO(model_path, verbose=False)

	for filename in os.listdir(ds_folder):

		input_path = os.path.join(ds_folder, filename)
		if not os.path.isfile(input_path) or not filename.endswith(('.mp4', '.avi')):
			continue

		# Open the video file
		cap = cv2.VideoCapture(input_path)
		if not cap.isOpened():
			continue

		frame_count = -1

		while cap.isOpened():
			frame_count += 1

			ret, frame = cap.read()
			if not ret:
				break

			# Infer every 5 frames
			if frame_count % 5:
				continue

			# Perform inference on the frame
			_, scores, boxes, _, _ = frame_inference(model, frame)

			# Collect the top 5 boxes
			for score, box in sorted(zip(scores, boxes), key=lambda t: t[0], reverse=True)[:4]:
				# filename_frame_x1_y1.jpg
				box_filename = f'{filename}_{frame_count}_{box.x1}_{box.y1}.jpg'
				box_filepath = os.path.join(gi_folder, box_filename)

				# Save the box image
				cv2.imwrite(box_filepath, frame[box.y1:box.y2, box.x1:box.x2])

		cap.release()

	return


def main_histogram(gi_folder: str, hue_bins: int = 180, sat_bins: int = 256):
	"""
	Generate the Hue-Saturation histogram for all images in the gi folder.
	Then print the distribution of the distance for each subfolder with cv2.compareHist(hist1, hist2, method).

	:param gi_folder: Path to the folder containing images.
	:param hue_bins: Number of bins for Hue.
	:param sat_bins: Number of bins for Saturation.
	:return: The averaged prototype histogram.
	"""

	# Circular imports baby!
	from ..model import FrameBox

	# Prototype histogram
	prototype_hist = compute_gi_histogram(hue_bins, sat_bins)

	if prototype_hist is None:
		print("No images found in the gi folder.")
		return

	# List of subfolders
	subfolders = ['white', 'blue', 'bin']

	# Loop through each subfolder
	for subfolder in subfolders:

		# Check if the subfolder exists
		subfolder_path = os.path.join(gi_folder, subfolder)
		if not os.path.exists(subfolder_path):
			continue

		scores = []

		# Loop through each image in the subfolder
		for filename in os.listdir(subfolder_path):
			file_path = os.path.join(subfolder_path, filename)
			if not os.path.isfile(file_path):
				continue

			img = cv2.imread(file_path)
			if img is None:
				continue

			# Compute image histogram
			box = FrameBox(0, 0, img.shape[1], img.shape[0])
			hist = box.histogram_huesat(img, hue_bins, sat_bins)

			# Compare histograms
			score = cv2.compareHist(hist, prototype_hist, cv2.HISTCMP_BHATTACHARYYA)
			scores.append(1 - score)

		if not scores:
			continue

		# Compute stats
		min_score = np.min(scores)
		avg_score = np.mean(scores)
		max_score = np.max(scores)
		median_score = np.median(scores)
		std_score = np.std(scores)

		# Print results
		print(f'\nSubfolder: {subfolder}')
		print(f'  Min: {min_score:.4f}')
		print(f'  Average: {avg_score:.4f}')
		print(f'  Max: {max_score:.4f}')
		print(f'  Median: {median_score:.4f}')
		print(f'  Std: {std_score:.4f}')

	return


def filename_gi_histogram(hue_bins: int, saturation_bins: int) -> str:
	"""
	Generate a filename for the histogram based on Hue and Saturation.

	:param hue_bins: Number of bins for Hue.
	:param saturation_bins: Number of bins for Saturation.
	:return: The filename for the histogram.
	"""

	return f'gi_hist_{hue_bins}_{saturation_bins}.npy'


def compute_gi_histogram(hue_bins: int, saturation_bins: int) -> np.ndarray:
	"""
	Creates a prototype Hue-Saturation histogram by averaging histograms from sample judogi images.

	:param hue_bins: Number of bins for Hue.
	:param saturation_bins: Number of bins for Saturation.
	:return: The averaged prototype histogram.
	"""

	# Get subfolders paths
	gi_folders = [
		os.path.join(MyEnv.dataset_source, 'gi', subfolder)
		for subfolder in ('white', 'blue')
	]

	# Get images paths
	images = [
		os.path.join(folder, filename)
		for folder in gi_folders
		for filename in os.listdir(folder)
		if filename.split('.')[-1] in {'jpg', 'png', 'jpeg'}
	]

	hists = []

	# Loop over all images
	for img_path in images:

		# Read image
		img = cv2.imread(img_path)
		if img is None:
			continue

		# Get hue and saturation channels
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hue = hsv_img[:, :, 0]
		saturation = hsv_img[:, :, 2]

		# Compute histogram
		hist = cv2.calcHist(
			[hue, saturation],
			[0, 1],
			None,
			[hue_bins, saturation_bins],
			[0, 180, 0, 256]
		)

		hists.append(hist)

	if not hists:
		raise FileNotFoundError("No images found in the gi folders to create a histogram.")

	# Compute average histogram (normalized)
	avg_hist = np.mean(hists, axis=0)
	cv2.normalize(avg_hist, avg_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

	return avg_hist
