// Firebase App (the core Firebase SDK) is always required and
// must be listed before other Firebase SDKs
////const firebase = require("firebase-admin");
const serviceAccount = require('lifit-98bf5-27cfdea7b5f8.json');

// Add the Firebase products that you want to use
require("firebase/auth");
require("firebase/firestore");

var firebaseConfig = {
    apiKey: "AIzaSyAKhN6UaV-mMfuFGF8Ut_YRSxgV_6fuDak",
    authDomain: "lifit-98bf5.firebaseapp.com",
    databaseURL: "https://lifit-98bf5.firebaseio.com",
    projectId: "lifit-98bf5",
    storageBucket: "lifit-98bf5.appspot.com",
    messagingSenderId: "141457582315",
    appId: "1:141457582315:web:8924908d678e547e50b6ae",
    measurementId: "G-8TQ1H65LM3"
  };
  // Initialize Firebase
//firebase.initializeApp(firebaseConfig);

const admin = require('firebase-admin');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();
let pico = "iffo";

getPost().then(theAi);


async function getPost() {

const postRef = db.collection('posts');
const querySnapshot = await postRef.orderBy('description', 'desc').limit(1).get()
if (querySnapshot.docs.length > 0) {
  const doc = querySnapshot.docs[0];
  console.log(doc.data().profileImage);
  pico = doc.data().profileImage;
} else {
  (err => {
    console.log('Error getting document', err);
});

}};


async function theAi() {

  const tf = require('@tensorflow/tfjs'),
  mobilenet = require('@tensorflow-models/mobilenet'),
  tfnode = require('@tensorflow/tfjs-node'),
  fs = require('fs-extra');

  
  tfimage = tfnode.node.decodeImage(pico),
  mobilenetModel = await mobilenet.load();  

  const results = await mobilenetModel.classify(tfimage); 

  console.log(results);


};