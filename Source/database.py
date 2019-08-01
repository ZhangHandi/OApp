import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('./oapp-ade18-firebase-adminsdk-t96wy-8a316b4cc8.json')
firebase_admin.initialize_app(cred, {'databaseURL': 'https://oapp-ade18.firebaseio.com/'})
# default_app = firebase_admin.initialize_app({'databaseURL': 'https://oapp-ade18.firebaseio.com/'})
#"/home/handi/Desktop/OApp_v0.1/source/oapp-ade18-firebase-adminsdk-t96wy-8a316b4cc8.json"
