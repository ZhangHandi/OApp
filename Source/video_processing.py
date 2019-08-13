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

def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """
    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break
            
        frame = video_getter.frame
        rgb_frame = frame[:, :, ::-1]
		# Find all the faces and face encodings in the current frame of video
        (top, right, bottom, left) = face_recognition.face_locations(rgb_frame)[0]
        face_height = bottom - top

		# Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

		# Display the resulting frame
        (x, y, w, h) = mouth_detection.mouth_detection_video(frame, detector, predictor)
        
        if h < 0.18*face_height:
            cv2.putText(frame, "close", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "open", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            roi = frame[y:y+h, x:x+w]
            (px, py, pw, ph) = utils.color_detection(roi)
            if pw != 0:
                cv2.rectangle(frame, (x+px, y+py), (x+px+pw, y+py+ph), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "no pill detected", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        video_shower.frame = frame
        fps.update()

fps = FPS().start()
threadVideoGet(args["video"])

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
