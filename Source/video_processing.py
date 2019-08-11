import cv2
import numpy as np
import time
import utils
import dlib
import face_recognition
import argparse
import mouth_detection

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args['video'])
# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
	ret, frame = cap.read()
	if ret:
		rgb_frame = frame[:, :, ::-1]

		# Find face in the current frame of video
		face_location = face_recognition.face_locations(rgb_frame)[0]

		# Label the results
		(top, right, bottom, left) = face_location
		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

		# Display the resulting frame
		(x, y, w, h) = mouth_detection.mouth_detection_video(frame, detector, predictor)

		# calculate face height
		face_height = bottom - top

		# we assume if mouth is less than 20% face height, then the mouth is close
		if h < 0.2*face_height:
			cv2.putText(frame, "close", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			cv2.putText(frame, "open", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# select mouth 
			roi = frame[y:y+h, x:x+w]
			(px, py, pw, ph) = utils.color_detection(roi)
			if pw != 0:
				cv2.rectangle(frame, (x+px, y+py), (x+px+pw, y+py+ph), (0, 255, 0), 2)
			else:
				cv2.putText(frame, "no pill detected", (50, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)		
		cv2.imshow('Frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

# When everything done, release the video capture object
cap.release()
cv2.destroyAllWindows()
