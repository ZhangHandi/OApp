import utils

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group2')

# load images to test
unknown_names, unknown_faces_encoding = utils.create_database('../Images/test_face/group2')

index = [utils.recognize_face(unknown_face_encoding, known_faces_encoding) for unknown_face_encoding in unknown_faces_encoding]
names = [known_names[i] for i in index]
utils.compare_result(unknown_names, names)

# results : 
# test only group 1 : Result : 2 error out of 18. Accuracy: 0.89
# test only group 2 : Result : 6 error out of 8. Accuracy: 0.25
# test group 3 (all photos) : Result : 9 error out of 26. Accuracy: 0.65