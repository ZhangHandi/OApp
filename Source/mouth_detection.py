import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np

def is_only_one_person(detect):
    if len(detect) != 1 :
        print('No person or not only one person in the image.')
        return False
    return True

def visualize_mouth(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
    if colors is None:
        colors = (19, 199, 109)

	# grab the (x, y)-coordinates associated with the
	# face landmark
    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    pts = shape[j:k]
    hull = cv2.convexHull(pts)
    cv2.drawContours(overlay, [hull], -1, colors, -1)
    
	# apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # return the output image
    return output


def mouth_location_image(image_path, predictor_path):
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

# function for video : input image
def mouth_detection_video(image, detector, predictor):
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
        
        # visualize_mouth(image, shape)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0))
        return x, y, w, h