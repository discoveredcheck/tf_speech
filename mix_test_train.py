import numpy as np

DATASET_DIR = '/home/guillaume/speech_dataset/'

if __name__ == '__main__':

    print('----------')
    test_data = np.load(DATASET_DIR + 'test/numpy/test_dataset_wsize50_wstride10_dct40_.npy')
    print('test:  ', test_data.shape)
    train_data = np.load(DATASET_DIR + 'train/numpy/train_dataset_wsize50_wstride10_dct40_.npy')
    print('train: ', train_data.shape)