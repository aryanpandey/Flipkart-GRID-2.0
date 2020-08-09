## Imports
import librosa
import os
import numpy as np

'''
Function Name: initialize_counter

Parameters: train_path - path to the directory with all the audio files for training
            target_path - path to the directory with all the audio files which are the targets for the training files

Returns: None

Function Description:
In the following function we initialize a few global variables
We get the names of the train files and valid files
We also make a dictionary to associate each train file to it's target file
This can be done pretty easily as long as the files are named properly
In the end we shuffle the file names so that the batches are not the same across passes
'''

def initialize_counter(train_path, target_path):
    #Initialize global variables
    global counter     #Variable for keeping track of how many files have been trained on in one pass
    global file_paths  #Dictionary to associate the train file to it's target
    global train_files #A numpy array which contains the file names from target_path to be used for training
    global valid_files #A numpy array which contains the file names from target_path to be used for validation
    
    #Get a sorted list of filenames in a directory
    train_files = np.sort(os.listdir(train_path))
    target_files = np.sort(os.listdir(target_path)) #A numpy array which contains the file_names from target_path to be used as a target for the training and validation data
    
    #Creating a dictionary to associate train files with their targets
    file_paths = {}
    
    for i in range(len(train_files)):
        file_paths[train_files[i]] = target_files[i]
    
    #Creating validation and train set
    valid_files = train_files[-200:]
    train_files = train_files[:-200]
    
    #Shuffling dataset
    np.random.shuffle(train_files)

    counter = 0

'''
Function Name: get_validation_set

Parameters: train_path - path to the directory with all the audio files for training
            target_path - path to the directory with all the audio files which are the targets for the training files
            A - Number of segments that an audio input will be split into
            L - Length of each audio segment

Returns: arr_pad_valid - A list of numpy arrays in which each numpy array represents the whole audio padded with zeros to a length of 10 seconds
         arr_pad_valid_target - A list of numpy arrays in which each numpy array represents the target audio padded with zeros to a length of 10 seconds
         arr_reshaped_valid - A list of numpy arrays in which each numpy array is a reshaped array which is fed into our neural network 

Function Description:
In the following function we iterate through the validation file names and for each file we perform some operations.
First the file is loaded into 'single_arr_valid'. After this it is padded with zeros at the end of the audio until it's length
becomes 10 seconds long. This is done to maintain a uniform length across all audio files.
This array is then reshaped to a shape of (1,A,L,1) and stored in 'single_arr_valid_reshaped'. This is the array that will be fed into the network.
We do the same thing for the target audio files and append them to their respective variables which will be returned by the function
'''

def get_validation_set(train_path, target_path, A, L):
    global valid_files
    global file_paths
    
    #Initializing a few empty lists to store the numpy arrays
    arr_pad_valid = []
    arr_pad_valid_target = []
    arr_reshaped_valid = []
    
    #Iterating through the validation files
    for f in valid_files:
        #Reading the data from each file at a sampling rate of 22000
        single_arr_valid, _ = librosa.load(train_path + f , sr=22000)
        #Padding the input array with zeros till it has a 10 second length
        single_arr_pad_valid = np.pad(single_arr_valid, (0, A*L - len(single_arr_valid)), 'constant', constant_values=(0,0))
        #Creating the reshaped array to feed into our network
        single_arr_reshaped_valid = single_arr_pad_valid.reshape(1, A, L, 1)
        
        #Doing the same for our target, only here we don't do any reshaping
        single_arr_target_valid, _ = librosa.load(target_path + file_paths[f] , sr=22000)
        single_arr_pad_target_valid = np.pad(single_arr_target_valid, (0, A*L - len(single_arr_target_valid)), 'constant', constant_values=(0,0))
        
        #Switching the dimension of the array from (A*L,1) to (1,A*L)
        single_arr_pad_valid = np.reshape(single_arr_pad_valid, (1, -1))
        #Appending the numpy array to the overall list
        arr_pad_valid.append(single_arr_pad_valid)
        
        #Doing the same thing for the target files
        single_arr_pad_target_valid = np.reshape(single_arr_pad_target_valid, (1, -1))
        arr_pad_valid_target.append(single_arr_pad_target_valid)
        
        #Appending the reshaped numpy arrays to the overall list
        arr_reshaped_valid.append(single_arr_reshaped_valid)
        
    #Reshaping the overall reshaped to (no. of files, Length, Breadth, Depth) format    
    arr_reshaped_valid = np.array(arr_reshaped_valid).reshape(200,A,L,1)
    
    #Reshaping the others to (no. of files, length of audio file) format
    arr_pad_valid = np.array(arr_pad_valid).reshape(200, A*L)
    arr_pad_valid_target = np.array(arr_pad_valid_target).reshape(200, A*L)
    
    return arr_pad_valid, arr_pad_valid_target, arr_reshaped_valid

'''
Function Name: inputProcess

Parameters: train_path - path to the directory with all the audio files for training
            target_path - path to the directory with all the audio files which are the targets for the training files
            A - Number of segments that an audio input will be split into
            L - Length of each audio segment
            batch_size - The batch size requested by the person training the network.

Returns: arr_pad_train - A list of numpy arrays in which each numpy array represents the whole audio padded with zeros to a length of 10 seconds
         arr_pad_target - A list of numpy arrays in which each numpy array represents the target audio padded with zeros to a length of 10 seconds
         arr_reshaped - A list of numpy arrays in which each numpy array is a reshaped array which is fed into our neural network 
         actual_batch_size - Returns the value of the actual batch size that was encountered in the function. This value might be different when the 
         number of files is not an integral multiple of the batch_size
         
Function Description:
In the following function we iterate through the train file names and for each file we perform some operations.
First the file is loaded into 'single_arr_train'. After this it is padded with zeros at the end of the audio until it's length
becomes 10 seconds long. This is done to maintain a uniform length across all audio files.
This array is then reshaped to a shape of (1,A,L,1) and stored in 'single_arr_reshaped'. This is the array that will be fed into the network.
We do the same thing for the target audio files and append them to their respective variables which will be returned by the function.
The counter is incremented by the actual batch size so that we can keep track of where we reached in the whole list of train file names.
'''

def inputProcess(train_path,target_path, A, L, batch_size):
    global counter
    global file_paths
    global train_files
    
    #Getting the required files for making the batch using the counter
    required_train = train_files[counter:counter+batch_size]
     
    #Initializing some empty lists to store the numpy arrays
    arr_pad_train = []
    arr_pad_target = []
    arr_reshaped = []
    
    #Iterating through the train files for this batch
    for f in required_train:
        
        #Loading, padding and reshaping the files as descriped in the previous function
        single_arr_train, _ = librosa.load(train_path + f , sr=22000)
        single_arr_pad_train = np.pad(single_arr_train, (0, A*L - len(single_arr_train)), 'constant', constant_values=(0,0))
        single_arr_reshaped = single_arr_pad_train.reshape(1, A, L, 1)
        
        #Doing the same for targets
        single_arr_target, _ = librosa.load(target_path + file_paths[f] , sr=22000)
        single_arr_pad_target = np.pad(single_arr_target, (0, A*L - len(single_arr_target)), 'constant', constant_values=(0,0))
        
        #Reshaping and appending
        single_arr_pad_train = np.reshape(single_arr_pad_train, (1, -1))
        arr_pad_train.append(single_arr_pad_train)
        
        single_arr_pad_target = np.reshape(single_arr_pad_target, (1, -1))
        arr_pad_target.append(single_arr_pad_target)
        
        arr_reshaped.append(single_arr_reshaped)
        
    #Getting the actual batch size and incrementing the counter accordingly
    actual_batch_size = len(required_train)
    counter = counter + actual_batch_size
    
    #Getting the reshaped list in the format (batch_size, length, breadth, depth)
    arr_reshaped = np.array(arr_reshaped).reshape(actual_batch_size,A,L,1)
    
    #Getting the rest of the lists in the format (batch_size, length of audio file)
    arr_pad_train = np.array(arr_pad_train).reshape(actual_batch_size, A*L)
    arr_pad_target = np.array(arr_pad_target).reshape(actual_batch_size, A*L)
    
    return arr_pad_train, arr_pad_target, arr_reshaped, actual_batch_size

'''
Function Name: inputProcesstest

Parameters: path - path to the file on which testing needs to be done
            A - Number of segments that an audio input will be split into
            L - Length of each audio segment
            
Returns: single_arr_pad - A padded numpy array which contains information about the test file
         single_arr_reshaped - A numpy array reshaped to (1,A,L,1) which will be passed to the network
         
Function Description:
This function takes in the path to a file, loads it into an array, pads it and then reshapes it.
This is done so that we can get the array to feed into the network to make predictions
'''

def inputProcesstest(path, A, L):
    #Load the file into the array, pad it and reshape it
    single_arr, _ = librosa.load(path, sr=22000)
    single_arr_pad = np.pad(single_arr, (0, A*L - len(single_arr)), 'constant', constant_values=(0,0))
    single_arr_reshaped = single_arr_pad.reshape(1, A, L, 1)
    
    #Switch the dimensions of the padded array
    single_arr_pad = np.reshape(single_arr_pad, (1, -1))
    return single_arr_pad, single_arr_reshaped

'''
Function Name: wavCreator

Parameters: path - A path to the place where the output file should be stored
            arr - The output array from the network after prediction
            
Returns: None

Function Description:
This function takes in the predicted array and writes it to the given path in wav format.
'''

def wavCreator(path, arr):
    arr = np.array(arr).T
    librosa.output.write_wav(path, arr, sr=22000)


