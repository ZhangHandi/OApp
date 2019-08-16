import os
import face_recognition
import cv2

def is_an_image_file(filename):
    '''
    Verfiy if the input file is an image file.
    :param filename: input filename
    :return: A boolean True or False
    '''
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    statinfo = os.stat(filename).st_size
    if (statinfo == 0):
        return False
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def load_paths(file_path):
    '''
    Verify each file in folder if it's an image file, and put all image files in list.
    :param file_path: path to input image folder
    :return: list contains all image path
    '''
    files = os.listdir(file_path)
    return [f for f in files if is_an_image_file(os.path.join(file_path, f))]

def load_paths_and_names(file_path):
    '''
    Load image paths and names of images
    :param file_path: path to input folder
    :return: image_paths: list of images paths, names: list of faces' names(name of image)
    '''
    files = os.listdir(file_path)
    paths = [f for f in files if is_an_image_file(os.path.join(file_path, f))]
    image_paths = [os.path.join(file_path, f) for f in paths]
    names = known_faces_name(paths)
    return image_paths, names

def known_faces_name(paths):
    '''
    Load all faces' names
    :param paths: image paths
    :return: all names in database
    '''
    known_names = []
    for path in paths:
        if path.endswith('.jpg') or path.endswith('.png'):
            known_names.append(path[:-4])
        elif path.endswith('.jpeg'):
            known_names.append(path[:-5])
    return known_names

def create_database(file_path):
    '''
    Create a datebase by loading images and transfer to face encodings
    :param file_path: path to image folder
    :return: known_names: list of names in database, face_images_encoding: list of face encodings
    for each image file
    '''
    paths = load_paths(file_path)
    known_names = known_faces_name(paths)
    image_paths = [os.path.join(file_path, f) for f in paths]
    face_images = [face_recognition.load_image_file(path) for path in image_paths]
    # get the face encodings for each image file
    # Given an image, return the 128-dimension face encoding for each face in the image.
    try:
        face_images_encoding = [face_recognition.face_encodings(face_image)[0] for face_image in face_images]
    except IndexError:
        print("Unable to locate any faces in at least one of the images.")
        quit()
    return known_names, face_images_encoding

def recognize_face(unknown_face_encoding, known_faces_encoding) :
    '''
    Recognize if it's the same face in database
    :param unknown_face_encoding: face encoding to be verified 
    :param known_faces_encoding: list of known face encoding
    :return: the index of face corresponded
    '''
    results = face_recognition.compare_faces(known_faces_encoding, unknown_face_encoding, tolerance=0.45)
    try:
        index = results.index(True)
        return index
    except:
        return len(known_faces_encoding)

def compare_result(expect_result, result):
    '''
    Compare each result for each face to be verified in folder
    :param expect_result: expect result for detected face
    :param result: actual result of detected face
    :return: result and accuracy
    '''
    count = 0
    for i in range(len(result)):
        if result[i] in expect_result[i]:
            count += 1
    return print("Result : %d error out of %d. Accuracy: %.2f" %(len(result) - count, len(result), count/len(result)))

def color_detection(image):
    '''
    Detect certain color in mouth
    :param image: input image
    :return: px, py, pw, ph: (x, y)-coordinate of top-left point, width and height of bounding box
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)

    mask = cv2.inRange(hsv, light_white, dark_white)
    result = cv2.bitwise_and(image, image, mask=mask)
    # threshold image
    ret, threshed_img = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY),
                    127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    px, py, pw, ph = 0, 0, 0, 0
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea) 
        # get the bounding rect
        px, py, pw, ph = cv2.boundingRect(c)
    return px, py, pw, ph




