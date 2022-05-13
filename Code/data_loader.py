import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader():
    def __init__(self, root_path='../', mode='hr'):
        self.root_path = root_path
        self.mode = mode

    def load_mths(self, dwn_factor=2, time_length=10, testset_size=0.2, validset_proportion=0.15):
        mred, mblue, mgreen = ([], [], [])
        hr, spo2 = ([], [])
        seq_len = int(30 / dwn_factor) * time_length
        print(f"Each sample's input signal sequence length would be {seq_len}")

        for i in range(67):
            signals, labels = None, None
            try:
                signals = np.load(self.root_path + '/MTHS/Data/signal_{}.npy'.format(i))
                labels = np.load(self.root_path + '/MTHS/Data//label_{}.npy'.format(i))
            except:
                continue

            mred.extend(signals[:, 0])
            mgreen.extend(signals[:, 1])
            mblue.extend(signals[:, 2])

            hr.extend(labels[:,0])
            spo2.extend(labels[:,1])
        
        # downsample
        mred, mblue, mgreen  = mred[::dwn_factor], mblue[::dwn_factor], mgreen[::dwn_factor]

        hr_t = np.zeros((len(mred)//seq_len,))
        spo2_t = np.zeros((len(mred)//seq_len,))
        ppg_t = None

        if self.mode == 'hr':
            ppg_t = np.zeros((len(mred)//seq_len, seq_len, 1)) # Only the red channel

        elif self.mode == 'spo2':
            ppg_t = np.zeros((len(mred)//seq_len, seq_len, 3)) # all three channels

        for i in range(ppg_t.shape[0]):
            hr_t[i] = np.mean(hr[i*time_length :  (i+1)*time_length])
            spo2_t[i] = np.mean(spo2[i*time_length :  (i+1)*time_length])

            start, end = i * seq_len, (i+1) * seq_len
            if self.mode=='hr':
                ppg_t[i,:,0] = mred[start : end]
            elif self.mode == 'spo2':
                ppg_t[i,:,0] = mred[start: end]
                ppg_t[i,:,1] = mblue[start : end]
                ppg_t[i,:,2] = mgreen[start : end]
        
        x_train, y_train, x_test, y_test = None, None, None, None
        if self.mode=='hr':
            x_train, x_test, y_train, y_test = train_test_split(ppg_t, hr_t, test_size=testset_size, random_state=1400) # RS must be specified**
        elif self.mode == 'spo2':
            x_train, x_test, y_train, y_test = train_test_split(ppg_t, spo2_t, test_size=testset_size, random_state=1400) # RS must be specified**
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validset_proportion, random_state=1400)
        
        print("X, y train shapes = ", (x_train.shape, y_train.shape))
        print("X, y valid shapes = ", (x_valid.shape, y_valid.shape))
        print("X, y test shapes = ", (x_test.shape, y_test.shape))

        return x_train, y_train, x_valid, y_valid, x_test, y_test