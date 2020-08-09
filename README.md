# Flipkart-GRID-Noise-Cancellation-Solution

This repository contains team Third Degree Burn's solution for Round 3 of the Flipkart GRiD 2.0.

## API Usage
### Without a GUI
We have provided an API which uses Flask to take in the path to an input file or a directory with multiple input files and the path to an output directory where the files will be stored in WAV format. To make use of our scripts, run [this](FlaskNoGUI/wsgi.py) wsgi script on a terminal and then run [this](FlaskNoGUI/testing.py) interacting script separately to start interacting with the server.

    $ cd FlaskNoGUI
    
    $ python wsgi.py

The scripts can be found here:
- Web Server Gateway Interface Script : [WSGI Script](FlaskNoGUI/wsgi.py)
- The Script for the App : [App Script](FlaskNoGUI/app.py) 
- Script for interacting with the server : [Script for Interacting](FlaskNoGUI/testing.py)

### With a GUI
We have also made the scripts for having a small GUI incorporated with Flask using tkinter. The order to run the scripts is the same. You first run [this](FlaskGUI/wsgi_GUI.py) wsgi script on a terminal and then run [this](FlaskGUI/testing_GUI.py) interacting script separately to start interacting with the server. The only difference here is that On running the interacting script, a pop-up window shows in which you can select whether you want to input a single file or a whole directory. In any case, you need to select the input directory before the output directory, else the code will not run.

The scripts for this can be found here:
- Web Server Gateway Interface Script : [WSGI Script](FlaskGUI/wsgi_GUI.py)
- The Script for the App : [App Script](FlaskGUI/app_GUI.py) 
- Script for interacting with the server : [Script for Interacting](FlaskGUI/testing_GUI.py)

## Model Building
Following are the scripts that we have used for building our model:
- Utility Script: [Utils](Scripts-with-batches/utils.py)
- Script which builds the Model: [Model](Scripts-with-batches/model.py)
- Main Script for training: [Training Script](Scripts-with-batches/flipkart-main.py)

## Datsets
We manually made our data where we created 30 odd files with just background noise and 30 odd files which contains clear voice. We then generated our dataset by mixing each clean audio with each background noise to create a new audio file. This way we got 1000 audio files. We also incorporated a function where we scaled our noise by some amount to get three sets of audio: One with a dimmed out noise, one with the noise at the same level as the clean audio and one with an amplified noise. Totally we got 3000 audio samples from this mixing.

The datasets as of now are uploaded to Kaggle and are Private. Anyone with the following links will be able to access it. The dataset will be made public after the competition is over, if Flipkart allows it.

- Link to the 3000+ Train Files: www.kaggle.com/dataset/41c6bd8cbc8109cdc51517df16c519d3f9ca3befc6e2f8d83a5595688b31caed
- Link to 3000+ Target Files: www.kaggle.com/dataset/6bbb0b80904bc32de86c41a8c537b2fa5cc955b3db00c3c5ac0559d2f4f34815
- Weights after Model was trained: www.kaggle.com/dataset/8d55d0a3859676a9c39f3ae29f33a7076e67168529a33aaeb85ff94e0bcde9ac
- Original Files from Flipkart: www.kaggle.com/dataset/10241db9506e5adeecabcfcd44c2ed3de5e79563e1473cc7eb3cea13b5644e47
