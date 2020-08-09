import requests
import json
from GUI import smallGUI
import os

headers = {'Content-type' : 'application/json'}
url = 'http://127.0.0.1:5000/predict'
#data = {'input_path' : 'D:\\Github Repos\\Flipkart-GRID-Noise-Cancellation2\\WAV-Inputs\\71.wav', 'output_path' : 'D:\\Github Repos\\Flipkart-GRID-Noise-Cancellation2\\FlaskAPI\\'}
guiobj = smallGUI()
outputdir = guiobj.outputdirpath + '/'

if(guiobj.check):
    data = {'input_path' : guiobj.inputfile, 'output_path' : outputdir + 'TDB_Prediction.flac'}
    res = requests.post(url, headers=headers, json=data)
    print(res) 

else:
    for music in sorted(os.listdir(guiobj.inputdirpath)):
        data = {'input_path' : guiobj.inputdirpath+ '/' + music, 'output_path' : outputdir + 'pred_' + music.split('.')[0] + '.flac'}
        res = requests.post(url, headers=headers, json=data)
        print(res)
