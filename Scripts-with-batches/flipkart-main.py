## Imports
import tensorflow as tf
import utils as utils
import os
from model import models
import numpy as np
import keras.backend as K

#Disabling eager execution
tf.compat.v1.disable_eager_execution()
#Clearing the keras backend session 
K.clear_session()

#Declaring all the variable parameters of our model
A = 2000    #Number of audio segments the file should be split into
L = 110     #Length of each audio segment
N = 512     #Number of basis vector (Encoded Length)
B = 128     #B and H are hyperparameters for the network
H = 512     
Sc = 128    #Number of Channels in the skip connection
vr = 3      #Number of TCNs (Temporal Convolution Network)
bl = 8

#Defining the models here
model = models(A, L, N, B, H, Sc, vr, bl)

#Loading the weights from our previous training into the model
model.gbl_model.load_weights('../input/weights-files/weights Pass-2(1).h5')

#Declaring the batch size
batch_size = 32

#Setting the paths for the training and target directories
train_path = '../input/flipkart-train-loud/Mixed/'
target_path = '../input/flipkart-target-loud/Target/'

#Iterating through the data multiple times
for i in range(5):
    #Here we intialize the counter to zero and set the train and validation file names
    utils.initialize_counter(train_path, target_path)
    
    #Extracting the required arrays for the validation data
    valid, valid_target, valid_reshaped = utils.get_validation_set(train_path, target_path, A, L)
    
    #Iterating through the training data
    for f in range(int(np.ceil((len(os.listdir(train_path)) - 200)/batch_size))):
        #Getting the required arrays for training
        in_arr_pad, out_arr_pad, in_arr_reshaped, updated_batch_size = utils.inputProcess(train_path, target_path, A, L, batch_size)
        
        #Sending in the data for training
        model.train(in_arr_reshaped, in_arr_pad, out_arr_pad, 10, 5, updated_batch_size, valid, valid_target, valid_reshaped)
    
    #Saving the weights after the pass
    model.gbl_model.save_weights('weights: Pass-' + str(i + 1) + '.h5')

#Predicting and writing on the first audio file from the data given to us
pred, to_pred = utils.inputProcesstest('../input/flipkart-round-3-original/0.mp3.wav', A, L)
predict = model.gbl_model.predict([to_pred, pred])
utils.wavCreator("0_output.wav", predict)

#Saving the overall model to use for deployment
model.gbl_model.save('gbl_model.h5')
