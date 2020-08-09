# -*- coding: utf-8 -*-
"""
Script for creating and loading contents to the server
"""
import flask
from flask import Flask, jsonify, request
import json
import tensorflow as tf
import librosa
import numpy as np
from scipy.io.wavfile import write
import tensorflow.keras.backend as K
import soundfile



def load_model():
    K.clear_session()
    model = tf.keras.models.load_model('Model/gbl_model.h5', compile=False)
    return model

def inputProcess(filepath, A=2000, L=110):
    arr, _ = librosa.load(filepath, sr=22000, duration=10)
    #arr = open(filepath, "r")
    #print("array = ",arr)

    arr_reshaped = arr.reshape(1, A, L, 1)

    return arr_reshaped

def wavCreator(path, arr):
    arr = np.array(arr).T
    #librosa.output.write_wav(path, arr, sr=22000)
    soundfile.write(path, arr, 22000)
    #write(path, 22000, arr)

app = Flask(__name__)

model = load_model()
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    response = json.dumps('')
    if request.method == 'POST':
        response = json.dumps('')
        print("Request: ", request)
        request_data = request.json
        #response = request_json
        #return response, 200
        print("request_data: ", request_data)
        #print("reached before file")
        filepath = request_data['input_path']
        #print("reached after file")
        path = request_data['output_path']
        print("path: ", path)
        arr_reshaped = inputProcess(filepath)
    
        
        denoised_arr = model.predict([arr_reshaped, np.zeros((1, 2000*110))])
    
        wavCreator(path, denoised_arr)
        
        response = json.dumps({1:2})

        return response, 200
    return response, 403

if __name__ == "__main__":
    app.run(debug=True)

