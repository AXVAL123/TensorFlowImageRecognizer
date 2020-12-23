import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import threading
import time

print('Initializing Firestore connection...')

cred = credentials.Certificate('lifit-98bf5-19cf51baee8a.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
callback_done = threading.Event()
print('Connection initialized')

# Create a callback on_snapshot function to capture changes
def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        print(f'Received document snapshot: {doc.id}')
    callback_done.set()

doc_ref = db.collection(u'postqueue')

# Watch the document
doc_watch = doc_ref.on_snapshot(on_snapshot)


