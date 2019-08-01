from google.cloud import storage
from firebase import firebase
import os
import utils

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/handi/Desktop/OApp_v0.1/source/oapp-ade18-firebase-adminsdk-t96wy-8a316b4cc8.json"
firebase = firebase.FirebaseApplication('https://oapp-ade18.firebaseio.com/')

client = storage.Client()
# the bucket name must not contain gs://
bucket = client.get_bucket('oapp-ade18.appspot.com')
image_blob = bucket.blob("/")

image_paths, names = utils.load_paths_and_names('../Images/database/group3')

for i in range(len(names)):
    image_blob = bucket.blob(names[i])
    image_blob.upload_from_filename(image_paths[i])