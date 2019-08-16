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
ap.add_argument("-d", "--database", required=True,
    help="path to input images file")
ap.add_argument("-n", "--name", required=True,
    help="patient's name")

args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# create database
known_names, known_faces_encoding = utils.create_database(args["database"])
known_names.append('unknown')

def thread_video(input):
    # Dedicated thread for grabbing video frames with VideoGet object.
    # Main thread shows video frames.
    video_getter = VideoGet(input).start() 

    nb_total = 0
    nb_fr = 0
    nb_pill = 0
    
    while True:
        if video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        rgb_frame = frame[:, :, ::-1]
    
        # find all the faces and make sure there can not be more than one person
        face_location = face_recognition.face_locations(rgb_frame)
        if len(face_location) == 0:
            pass
        elif len(face_location) > 1:
            print("WARNING: two person appear!")
            pass
        else:
            unknown_face_encoding = face_recognition.face_encodings(rgb_frame, face_location)[0]
            index = utils.recognize_face(unknown_face_encoding, known_faces_encoding)
            name = known_names[index]
            if name == args["name"]:
                nb_fr += 1
            try:
                top, right, bottom, left = face_location[0]
                face_height = bottom - top
                (x, y, w, h) = mouth_detection.mouth_detection_video(frame, detector, predictor)
                if h > 0.2*face_height:
                    d = int(0.35*h)
                    roi = frame[y+d:y+h, x:x+w]
                    (px, py, pw, ph) = utils.color_detection(roi)
                    if pw != 0:
                        nb_pill += 1
            except:
                        pass
            fps.update()
            nb_total += 1    
    return nb_total, nb_fr, nb_pill

fps = FPS().start()
nb_total, nb_fr, nb_pill = thread_video(args["video"])
fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if nb_pill/nb_total < 0.15:
    print("[INFO] no pill detected : {:.2f}".format(nb_pill/nb_total))
else:
    print("[INFO] pill detected : {:.2f}".format(nb_pill/nb_total))

if nb_fr/nb_total > 0.6:
    print("[INFO] right person : {:.2f}".format(nb_fr/nb_total))
else:
    print("[INFO] wrong person : {:.2f}".format(nb_fr/nb_total))