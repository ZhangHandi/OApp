import utils
import mouth_detection
import os

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group1')

# load images to test
unknown_names, unknown_faces_encoding = utils.create_database('../Images/test_face/group1')
known_names.append('unknown')

index = [utils.recognize_face(unknown_face_encoding, known_faces_encoding) for unknown_face_encoding in unknown_faces_encoding]
names = [known_names[i] for i in index]
utils.compare_result(unknown_names, names)

# results : 
# test only group 1 : Result : 1 error out of 18. Accuracy: 0.94
# test only group 2 : Result : 0 error out of 8. Accuracy: 1
# test group 3 (all photos) : Result : 1 error out of 26. Accuracy: 0.96

filepath = '../Images/test_mouth/'
predictor_path = './shape_predictor_68_face_landmarks.dat'

paths = utils.load_paths(filepath)
image_paths = [os.path.join(filepath, f) for f in paths]

for image_path in image_paths:
    mouth_detection.mouth_location_image(image_path, predictor_path)
