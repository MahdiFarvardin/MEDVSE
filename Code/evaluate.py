import tensorflow as tf
import argparse

from data_loader import DataLoader
import plot_results


# ----------------- Define cml arguments ---------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, default='hr', help='The training mode (hr or spo2)')
parser.add_argument('--dataset',type=str, required=False, default='mths', help='The name of the dataset to train on.')
parser.add_argument('--downsample',type=str, required=False, default=2, help='The down sample factor for ppg signals')
parser.add_argument('--timelen',type=str, required=False, default=10, help='time length of each input signal to the neural net')
parser.add_argument('--batchsize',type=str, required=False, default=32, help='batch size')
parser.add_argument('--testsize',type=str, required=False, default=0.2, help='testset proportion (of the whole data)')
parser.add_argument('--valsize',type=str, required=False, default=0.15, help='validation set proportion (of the train data)')
parser.add_argument('--savedir',type=str, required=False, default='./', help='Specifies a directory to save the trained models')

# ------------- Parse cml arguments and set config ------------------ #
args = parser.parse_args()
TRAIN_MODE = args.mode
DATASET_NAME = args.dataset
DOWNSAMPLE_FACTOR = args.downsample
TIME_LENGTH = args.timelen
TEST_SIZE = args.testsize
VALID_SIZE = args.valsize
saving_dir = args.savedir
SEQ_LEN = int(30 / DOWNSAMPLE_FACTOR) * TIME_LENGTH

BATCH_SIZE = args.batchsize

if __name__=='__main__':
    assert TRAIN_MODE in ['hr', 'spo2'], "The training mode must be one of 'hr' or 'spo2'."
    # Load the dataset
    loader = DataLoader(mode=TRAIN_MODE) # HR or Spo2
    x_train, y_train, x_valid, y_valid, x_test, y_test = None, None, None, None, None, None
    if DATASET_NAME=='mths':
        x_train, y_train, x_valid, y_valid, x_test, y_test = loader.load_mths(dwn_factor=DOWNSAMPLE_FACTOR, time_length=TIME_LENGTH, testset_size=TEST_SIZE, validset_proportion=VALID_SIZE)
    else:
        print("Wrong dataset name - Please check and try again")
        exit()

    model_names = ['BASE','FCN','FCN_Residual', 'FCN_Novel', 'FCN_DCT']
    losses = ['huber_loss','mean_squared_error','mean_absolute_error','log_cosh']
    model_list = []
    results = []
    
    for name in model_names:
        for loss in losses:
            model_list.append(tf.keras.models.load_model(saving_dir+'/{}_{}_{}_{}.h5'.format(DATASET_NAME, TRAIN_MODE, name, loss)))
            print("Evaluating {} with {}".format(name, loss))
            results.append(model_list[-1].evaluate(x_test, y_test , verbose=1))
            print()
    
    # Plot all the results .............
    # plot_results.plot(results)