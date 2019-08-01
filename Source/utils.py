import os
import face_recognition

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    statinfo = os.stat(filename).st_size
    if (statinfo == 0):
        return False
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def load_paths(file_path):
    files = os.listdir(file_path)
    return [f for f in files if is_an_image_file(os.path.join(file_path, f))]

def load_paths_and_names(file_path):
    files = os.listdir(file_path)
    paths = [f for f in files if is_an_image_file(os.path.join(file_path, f))]
    image_paths = [os.path.join(file_path, f) for f in paths]
    names = known_faces_name(paths)
    return image_paths, names

def known_faces_name(paths):
    known_names = []
    for path in paths:
        if path.endswith('.jpg') or path.endswith('.png'):
            known_names.append(path[:-4])
        elif path.endswith('.jpeg'):
            known_names.append(path[:-5])
    return known_names

def create_database(file_path):
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
    results = face_recognition.compare_faces(known_faces_encoding, unknown_face_encoding, tolerance=0.4)
    try:
        index = results.index(True)
        return index
    except:
        return len(known_faces_encoding)

def compare_result(expect_result, result):
    count = 0
    for i in range(len(result)):
        if result[i] in expect_result[i]:
            count += 1
    return print("Result : %d error out of %d. Accuracy: %.2f" %(len(result) - count, len(result), count/len(result)))






