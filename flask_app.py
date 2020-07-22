
# A very simple Flask Hello World app for you to get started with...

from flask import *
from datetime import *
from pytz import *

def get_pst_time():
    date_format='%m/%d/%Y %H:%M:%S'
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime=date.strftime(date_format)
    return pstDateTime


import os

app = Flask(__name__)

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.models import load_model as load
import cv2
import numpy as np
import h5py

@app.route('/upload')
def upload():
    return render_template("up.html")

@app.route('/upload-malaria')
def uploadMalaria():
    return render_template("index3.html")

@app.route('/upload-skin-cancer')
def uploadSkin():
    return render_template("up2.html")

@app.route('/upload-glaucoma')
def uploadGlaucoma():
    return render_template("glaucoma.html")

@app.route('/upload-brain-tumors')
def uploadBrain():
    return render_template("brain.html")

@app.route('/upload-tuberculosis')
def uploadTB():
    return render_template("tb.html")

diagnoses = []

@app.route('/past')
def go():
    ret = ""
    for item in diagnoses:
        ret = ret + item + "\n\n"
    return ret

@app.route('/glaucoma', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        h5file =  "/home/topdoc/mysite/dataweightsvha.h5"
        with h5py.File(h5file,'r') as fid:
            model = load_model(fid)
            Class = prediction(model, f.filename)
            diagnoses.clear()
            if (Class == 0):
                today = str(get_pst_time())
                diagnoses.append(today + ": According to our algorithm, it is likely that you have glaucoma. Please contact a medical professional as soon as possible for advice.")
                return f.filename + ": It is likely that you have glaucoma. Please contact a medical professional as soon as possible for advice."
            else:
                today = str(get_pst_time())
                diagnoses.append(today + ": According to our algorithm, you do not have glaucoma! If you have further questions, please contact a medical professional.")
                return f.filename + ": According to our algorithm, you do not have glaucoma! If you have further questions, please contact a medical professional."

@app.route('/malaria', methods = ['POST'])
def successMalaria():
    if request.method == 'POST':
        f2 = request.files['file']
        f2.save(f2.filename)
        h5file2 =  "/home/topdoc/mysite/prunedWeights2.h5"
        with h5py.File(h5file2,'r') as fid:
            model2 = load(fid)
            Class = prediction2(model2, f2.filename)
            diagnoses.clear()
            if (Class == 1):
                today = str(get_pst_time())
                diagnoses.append(today + ": According to our algorithm, it is likely that you have malaria. Please contact a medical professional as soon as possible for advice.")
                print(Class)
                return f2.filename + ": It is likely that you have malaria. Please contact a medical professional as soon as possible for advice."
            else:
                today = str(get_pst_time())
                diagnoses.append(today + ": According to our algorithm, you do not have malaria! If you have further questions, please contact a medical professional.")
                return f2.filename + ": According to our algorithm, you do not have malaria! If you have further questions, please contact a medical professional."

@app.route('/skin', methods = ['POST'])
def successSkin():
    if request.method == 'POST':
        f3 = request.files['file']
        f3.save(f3.filename)
        h5file3 =  "/home/topdoc/mysite/weights2.h5"
        with h5py.File(h5file3,'r') as fid:
            model3 = load(h5file3)
            finalPrediction = prediction3(model3, f3.filename)
            ret = ""
            if (finalPrediction == 0):
                ret = "You have been diagnosed with Melanocytic nevi. Please contact a doctor for assistance soon."
            elif (finalPrediction == 1):
                ret = "You have been diagnosed with Melanoma. Please contact a doctor for assistance soon."
            elif (finalPrediction == 2):
                ret = "You have been diagnosed with benign keratosis-like lesions, which is not a form of skin cancer. However, to be sure, please contact a doctor soon to confirm."
            elif (finalPrediction == 3):
                ret = "You have been diagnosed with Basal Cell Carcinoma. Please contact a doctor for assistance soon."
            elif (finalPrediction == 4):
                ret = "You have been diagnosed with Actinic Keratoses. Please contact a doctor for assistance soon."
            elif (finalPrediction == 5):
                ret = "You have been diagnosed with Vascular Lesions. Please contact a doctor for assistance soon."
            else:
                ret = "You have been diagnosed with Dermatofibroma. Please contact a doctor for assistance soon."
            diagnoses.clear()
            today = str(get_pst_time())
            diagnoses.append(today + ": " + ret)
            return f3.filename + ": " + ret

@app.route('/brain', methods = ['POST'])
def successBrainTumor():
    if request.method == 'POST':
        f4 = request.files['file']
        f4.save(f4.filename)
        h5file4 =  "/home/topdoc/mysite/VGGModel.h5"
        with h5py.File(h5file4,'r') as fid:
            model4 = load(fid)
            Class = prediction4(model4, f4.filename)
            diagnoses.clear()
            if (Class < 0.5):
                ret = "Congratulations! You are healthy! If you have further questions, please contact a medical professional."
            else:
                ret = "Unfortunately, you have been diagnosed with a Brain Tumor. Consider using our Tumor Detection Algorithm to better understand your diagnosis."
            diagnoses.clear()
            today = str(get_pst_time())
            diagnoses.append(today + ": " + ret)
            return f4.filename + ": " + ret

@app.route('/tb', methods = ['POST'])
def successTB():
    if request.method == 'POST':
        f5 = request.files['file']
        f5.save(f5.filename)
        h5file5 =  "/home/topdoc/mysite/TBModel.h5"
        with h5py.File(h5file5,'r') as fid:
            model5 = load(fid)
            Class = prediction5(model5, f5.filename)
            diagnoses.clear()
            if (Class == 0):
                ret = "Congratulations! You are healthy! If you have further questions, please contact a medical professional."
            else:
                ret = "Unfortunately, you have been diagnosed with Tuberculosis."
            diagnoses.clear()
            today = str(get_pst_time())
            diagnoses.append(today + ": " + ret)
            return f5.filename + ": " + ret

def autoroi(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = img[y:y+h, x:x+w]

    return roi

def prediction2(m, file):
    # list_of_files = glob.glob('data/test/*')
    # latest_file = max(list_of_files, key=os.path.getctime)
    img = cv2.imread(file)
    img = autoroi(img)
    img = cv2.resize(img, (125, 125))
    img = np.reshape(img, [1, 125, 125, 3])
    img = tf.cast(img, tf.float64)

    Class = m.predict(img)
    # Class = prob.argmax(axis=-1)

    return(Class)

def prediction3(m, file):
    img = cv2.imread(file)
    img = autoroi(img)
    img = cv2.resize(img, (75, 100))
    img = np.reshape(img, [1, 75, 100, 3])
    img = tf.cast(img, tf.float64)

    prediction = m.predict(img)
    prediction = prediction.argmax(axis=1)

    return(prediction)

def prediction(m, file):
    # list_of_files = glob.glob('data/test/*')
    # latest_file = max(list_of_files, key=os.path.getctime)
    img = cv2.imread(file)
    img = autoroi(img)
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, [1, 256, 256, 3])

    Class = m.predict(img)
    Class = prob.argmax(axis=-1)

    return(Class)

def prediction4(m, file):
    img = cv2.imread(file)
    img = autoroi(img)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    img = tf.cast(img, tf.float64)

    prediction = m.predict(img)
    prediction = prediction.argmax(axis=1)

    return(prediction)

def prediction5(m, file):
    img = cv2.imread(file)
    img = autoroi(img)
    img = cv2.resize(img, (96, 96))
    img = np.reshape(img, [1, 96, 96, 3])
    img = tf.cast(img, tf.float64)

    prediction = m.predict(img)
    Class = prediction.argmax(axis=1)

    return(Class)
