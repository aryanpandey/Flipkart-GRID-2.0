import requests
import json
import os

headers = {'Content-type' : 'application/json'}
url = 'http://127.0.0.1:5000/predict'
#data = {'input_path' : 'D:\\Github Repos\\Flipkart-GRID-Noise-Cancellation2\\WAV-Inputs\\71.wav', 'output_path' : 'D:\\Github Repos\\Flipkart-GRID-Noise-Cancellation2\\FlaskAPI\\'}
datatemp = {'input_path': input("Input filepath or path of directory with multiple files :"), 'output_path': input("Output directory :")}
isFile = os.path.isfile(datatemp['input_path'])
data = {}
if(isFile):
    data = datatemp
    data['output_path'] = data['output_path']+'\TDB_Prediction.flac'
    res = requests.post(url, headers=headers, json=data)
    print(res)

else:
    for music in os.listdir(datatemp['input_path']):
        data['input_path'] = datatemp['input_path'] + '\\' + music
        data['output_path'] = datatemp["output_path"] + '\pred_' + music
        res = requests.post(url, headers=headers, json=data)
        print(res)

