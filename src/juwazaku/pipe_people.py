import cv2
import mediapipe as mp
import torch


def main():
	# Model
	yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

	# since we are only intrested in detecting person
	yolo_model.classes = [0]

	mp_drawing = mp.solutions.drawing_utils
	mp_pose = mp.solutions.pose

	video_path = "C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\out\\LIS2104D1T1-8220-8340.mp4"

	# get the dimension of the video
	cap = cv2.VideoCapture(video_path)
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	size = (w, h)
	print(f"Size: {size}")

	cap = cv2.VideoCapture(video_path)

	# For saving the video file as output.avi
	out_path = "C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\out\\LIS2104D1T1-8220-8340-pose.mp4"
	out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MP4V"), cap.get(cv2.CAP_PROP_FPS), size)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		# Recolor Feed from RGB to BGR
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# making image writeable to false improves prediction
		image.flags.writeable = False

		result = yolo_model(image)

		# Recolor image back to BGR for rendering
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# we need some extra margin bounding box for human crops to be properly detected
		MARGIN = 10

		for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():

			# at least one size must be 10% of the frame
			if (xmax - xmin) < 0.08 * w or (ymax - ymin) < 0.08 * h:
				continue

			# Draw bounding box
			cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

			with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
				# Media pose prediction ,we are
				results = pose.process(
					image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:])

				# Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing
				mp_drawing.draw_landmarks(
					image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
					results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
					mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
					mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
				)

		# Write the frame into the output video
		out.write(image)

		# Code to quit the video incase you are using the webcam
		cv2.imshow('Activity recognition', image)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
