import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np

def is_only_one_person(detect):
    '''
    This function will detect if there's is only one person in picture by verifying length of 
    input list which contains all the face locations detected.
    :param detect: A list of face locations detected
    :return: A boolean True or False
    WARNING : Due to the fact that face locations will sometimes detected by mistake(i.e only one person in image
    but detected zero or two), so this function will not be used.
    '''
    if len(detect) != 1 :
        print("no person or not only one person in image.")
        return False
    return True

def visualize_mouth(image, shape, colors=None, alpha=0.75):
    '''
    :param immage: input image
    :parma shape: a NumPy array which contains (x, y)-coordinates of all the facial landmarks for the face region
    :param colors: unique color for visualize facial landmark region, None by default 
    :param alpha: weight of one copy of the input image(variable overlay), 0.75 by default
    :return: output image with mouth shown
    '''
	# create two copies of the input image -- one for the overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

	# if the colors list is None, initialize it with a unique color for each facial landmark region
    if colors is None:
        colors = (19, 199, 109)

	# grab the (x, y)-coordinates associated with mouth
    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    pts = shape[j:k]
    hull = cv2.convexHull(pts)
    cv2.drawContours(overlay, [hull], -1, colors, -1)
    
	# apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output


def mouth_location_image(image_path, predictor_path):
    '''
    Detect mouth location in image then visualize it. This function was used for only image processing(main_image.py
    in this project)
    :param image_path: image path to load image
    :param predictor_path: path to load the facial landmark predictor
    '''
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # load the input image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rect = detector(gray, 1)
    if not is_only_one_person(rect):
        quit()
    else:
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect[0])
        shape = face_utils.shape_to_np(shape)
    
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()
        cv2.putText(clone, "mouth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

        # loop over the subset of facial landmarks, drawing the
        # specific face part
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        mouth = image[y:y + h, x:x + w]
        mouth = imutils.resize(mouth, width=250, inter=cv2.INTER_CUBIC)

        # show the particular face part
        cv2.imshow("mouth", mouth)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)   

        # visualize all facial landmarks with a transparent overlay
        output = visualize_mouth(image, shape)
        cv2.imshow("Image", output)
        cv2.waitKey(0)     

def mouth_detection_video(image, detector, predictor):
    '''
    Same function as mouth_detection image, except detect mouth location in video then no need to visualize it. 
    This function was used for only video processing(main_video.py in this project)
    :param image: video frame 
    :param detector: dlib's face detector (HOG-based)
    :param predictor: the facial landmark predictor
    :return: x, y, w, h: (x, y)-coordinates of the left-top point of mouth bounding box
    w: width of bounding box, h: height of bounding box
    '''
    # load the input image and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    rect = detector(gray, 1)
    if not is_only_one_person(rect):
        pass
    else:
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect[0])
        shape = face_utils.shape_to_np(shape)
    
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0))
        return x, y, w, h