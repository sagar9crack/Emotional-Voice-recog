import numpy as np
import keras
import time
import librosa
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import sys
import librosa
import bulkDiarize as bk
model = keras.models.load_model('model/lstm_cnn_rectangular_lowdropout_trainedoncustomdata.h5')

classes = ['Calm', 'Passionate', 'Low-Confidence',
           'Aggresive', 'Too-Low-Confidence', 'Disgusted', 'Surprised']


def predict(folder, classes, model):
    solutions = []
    filenames=[]
    for subdir in os.listdir(folder):
        # print(subdir)
        
        lst = []
        predictions=[]
        # print("Sub",subdir)
        filenames.append(subdir)
        for file in os.listdir(f'{folder}{"/"}{subdir}'):
            # print(subdir,"+",file)
            temp = np.zeros((1,13,216))
            X, sample_rate = librosa.load(os.path.join(f'{folder}{"/"}{subdir}{"/"}', file), res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
            result = np.zeros((13,216))
            result[:mfccs.shape[0],:mfccs.shape[1]] = mfccs
            temp[0] = result
            t = np.expand_dims(temp,axis=3)
            ans=model.predict_classes(t)
            # print("SOL",classes[ans[0]])
            predictions.append(classes[ans[0]])

        if len(predictions) < 2:
            predictions.append('None')    
        solutions.append(predictions)
    return solutions,filenames


if __name__ == '__main__':
    INPUT_FOLDER_PATH = "input/"
    OUTPUT_FOLDER_PATH = "output/"
    # bk.diarizeFromFolder(INPUT_FOLDER_PATH,OUTPUT_FOLDER_PATH)
    for subdir in os.listdir(INPUT_FOLDER_PATH):
        bk.diarizeFromFolder(f'{INPUT_FOLDER_PATH}{subdir}{"/"}',(f'{OUTPUT_FOLDER_PATH}{subdir}{"/"}'))
        print("Diarized",subdir)



    folder = OUTPUT_FOLDER_PATH
    for subdir in os.listdir(folder):
        predictions,filenames = predict(f'{folder}{"/"}{subdir}', classes, model)
        # print("filename:",filenames,",Predictions:",predictions)
        with open('SER_'+subdir+'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            for i in range(len(filenames)):
                csvData = [filenames[i], 'person01',predictions[i][0],'person02',predictions[i][1]]
                print("filename:",filenames[i],",Predicted Emotion := Person1:",predictions[i][0],",Person2:",predictions[i][1])
                writer.writerow(csvData)
        csvFile.close()
    os.remove("filterTemp.wav")
                                        
                                        
   mylist= os.listdir('/content/Emotional-Voice-recog/src/Emotional-Voice-recog/src/Emotional-Voice-recog/Emotional-Voice-recog/src')
           type(mylist)
           print(mylist[1800])  
           print(mylist[400][6:-16])
 
 data, sampling_rate = librosa.load('INPUT_FOLDER_PATH')
% pylab inline
import os
import pandas as pd
import librosa
import glob 

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys


sr,x = scipy.io.wavfile.read('INPUT_FOLDER_PATH')

## Parameters: 10ms step, 30ms window
nstep = int(sr * 0.01)
nwin  = int(sr * 0.03)
nfft = nwin

window = np.hamming(nwin)

## will take windows x[n1:n2].  generate
## and loop over n2 such that all frames
## fit within the waveform
nn = range(nwin, len(x), nstep)

X = np.zeros( (len(nn), nfft//2) )

for i,n in enumerate(nn):
    xseg = x[n-nwin:n]
    z = np.fft.fft(window * xseg, nfft)
    X[i,:] = np.log(np.abs(z[:nfft//2]))

plt.imshow(X.T, interpolation='nearest',
    origin='lower',
    aspect='auto')

plt.show()
