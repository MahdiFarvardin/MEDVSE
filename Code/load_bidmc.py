# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

# It might take some time to download.
# !wget https://physionet.org/static/published-projects/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0.zip -q
# !unzip -q ./bidmc-ppg-and-respiration-dataset-1.0.0.zip

def load_bidmc(path='./bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv/', downsample_factor=8, conseq_seq=10):

  """
    conseq_seq means we need to have ppg samples with 10 seconds length
    we use downsample_factor to downsample 125(ish) hz to specified factor (e.g. 125 / 8 = 15.6)

    For example if you set downsample factor to 8, then we have a ppg signal of 15.6 sample per seconds. by setting conseq_seq to 10 we would have samples with length of 156.  

  """

  ages = np.zeros((53,))
  genders = np.zeros((53,), dtype=bool)
  input_seq_len = int(np.round(conseq_seq * (125 / downsample_factor)))
  X = np.zeros((53 , (480 // conseq_seq), input_seq_len, 1), dtype=np.float32)
  y = np.zeros((53 * (480 // conseq_seq) , 2), dtype= np.float32)

  for i in range(1,54):
    signals = pd.read_csv(path + '/bidmc_{:02d}_Signals.csv'.format(i))
    numerics = pd.read_csv(path + '/bidmc_{:02d}_Numerics.csv'.format(i))

    picked_num = pd.DataFrame(numerics, columns=[' HR',' SpO2'], copy=True)
    picked_signals = pd.DataFrame(signals, columns=[' PLETH'], copy=True)

    with open(path + '/bidmc_{:02d}_Fix.txt'.format(i)) as f:
      lines = f.readlines()
      try:
        ages[i-1] = int(lines[5][-3:-1])
        if lines[6][-2] == 'F':
          genders[i-1] = True
      except:
        ages[i-1] = 90

    xt = picked_signals.to_numpy()[::downsample_factor][:-1]
    xt = np.array_split(xt.ravel(), np.round(xt.shape[0] / input_seq_len))

    for j, x in enumerate(xt):
      if x.shape[0] != input_seq_len:
        #print("warning!")
        x = x[:input_seq_len]
      X[(i-1), j, :, 0] = x

    lbl = np.array_split(picked_num.to_numpy()[:480], 480//conseq_seq)
    mean_lbl = [[np.mean(a[:,0]), np.mean(a[:,1])] for a in lbl]
    #y.append(np.asarray(mean_lbl))
    y[(i-1)*(480 // conseq_seq): (i)*(480 // conseq_seq), :] = np.asarray(mean_lbl)

  return X.reshape((-1,input_seq_len)), y, ages, genders


ppg_signal, hr_spo2_label, ages, genders = load_bidmc()