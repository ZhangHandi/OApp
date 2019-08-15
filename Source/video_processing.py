# import the necessary packages
import cv2
import argparse
import numpy as np
import time
import utils
import dlib
import face_recognition
import mouth_detection
from VideoGet import VideoGet
from VideoShow import VideoShow
from imutils.video import FPS

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group3')
known_names.append('unknown')

def thread_video(input):
	"""
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
	"""

	video_getter = VideoGet(input).start()
	video_shower = VideoShow(video_getter.frame).start()	

	while True:
		if video_getter.stopped or video_shower.stopped:
			video_shower.stop()
			video_getter.stop()
			break

		frame = video_getter.frame
		rgb_frame = frame[:, :, ::-1]

		# Find all the faces and face encodings in the current frame of video
		face_location = face_recognition.face_locations(rgb_frame)
		if len(face_location) == 0:
			pass
		elif len(face_location) > 1:
			pass
		else:
			unknown_face_encoding = face_recognition.face_encodings(rgb_frame, face_location)[0]
			index = utils.recognize_face(unknown_face_encoding, known_faces_encoding)
			name = known_names[index]
			cv2.putText(frame, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		top, right, bottom, left = face_location[0]
		face_height = bottom - top

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

		# Display the resulting frame
		#try:
		(x, y, w, h) = mouth_detection.mouth_detection_video(frame, detector, predictor)

		if h < 0.2*face_height:
			cv2.putText(frame, "close", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			cv2.putText(frame, "open", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			d = int(0.35*h)
			roi = frame[y+d:y+h, x:x+w]
			#cv2.rectangle(frame, (x, y + int(0.2*h)), (x+w, y+h), (0, 255, 0), 2)
			(px, py, pw, ph) = utils.color_detection(roi)
			if pw != 0:
				cv2.rectangle(frame, (x+px, y+d+py), (x+px+pw, y+d+py+ph), (0, 255, 0), 2)
			else:
				cv2.putText(frame, "no pill detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		#except:
		#	pass
		video_shower.frame = frame
		fps.update()

fps = FPS().start()
thread_video(args["video"])

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
