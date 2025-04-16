import os

import cv2
from ultralytics import YOLO


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
