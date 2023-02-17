import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

"""
    This script is published only to address how we syncronized the mean RGB signals of each video with the labels
    Since in some patients the recorded video time was longer than its lables or vice versa.
    The initial .xslx files are not provided to the public since it is not necessary. 
    The npy labels should be used for the synchronization based on their lenght.
    If there are 65 samples on the npy label file, then you are going to limit the mean signals time to 65 seconds equivalent to 65*30 samples.
"""

# iterate over patient measurements
for i in range(1,67):
    
    print("Processing file #{i}")
    # Read excel labels (note: our labels was first recorded in an excel sheet)
    labels = pd.read_excel('./{}/labels.xlsx'.format(i), sheet_name='Sheet1')

    # load the mean RGB signals for the current video 
    # Note: These mean rgb signals were simply calculated from the raw videos.
    with open('./{}/signals.pkl'.format(i), 'rb') as pkl:
        signals = pickle.load(pkl)
    
    # Separate each channel of the data
    mRed = [signal[0] for signal in signals]
    mGreen = [signal[1] for signal in signals]
    mBlue = [signal[2] for signal in signals]

    # Here the cutting time stamp is determined --> this is done by checking which of the signal or the label has the lower length
    # and taking that min as the cutting time stamp (note that the video is recorded as 30 fps)
    cutting_time_stamp = min([labels['HR'].values.shape[0], (len(mRed)//30)])
    print(f"The cutting timestamp is {cutting_time_stamp}s.")

    # Now we know the cutting time stamp --> just create numpy arrays for the new cutted signal and labels
    signal = np.zeros((cutting_time_stamp*30, 3))
    lbl = np.zeros((cutting_time_stamp, 2))
    print(lbl.shape)

    lbl[:,0] = labels['HR'].values[:cutting_time_stamp]
    lbl[:,1] = labels['SPO2'].values[:cutting_time_stamp]

    signal[:, 0] = mRed[:cutting_time_stamp*30]
    signal[:, 1] = mGreen[:cutting_time_stamp*30]
    signal[:, 2] = mBlue[:cutting_time_stamp*30]
    '''
    plt.plot(signal[:,0])
    plt.plot(lbl[:,0])
    plt.plot(lbl[:,1])
    plt.show()
    '''

    # save them
    np.save('./final_data/signal_{}.npy'.format(i), signal)
    np.save('./final_data/label_{}.npy'.format(i), lbl)