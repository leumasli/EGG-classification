import numpy as np
from scipy import signal


def load_data(path="/content/drive/Shareddrives/EE147/project_data/", verbose=True):
    print("Loading data from", path)
    X_train_valid = np.load(path+"X_train_valid.npy")
    y_train_valid = np.load(path+"y_train_valid.npy")
    person_train_valid = np.load(path+"person_train_valid.npy").flatten()

    X_test = np.load(path+"X_test.npy")
    y_test = np.load(path+"y_test.npy")
    person_test = np.load(path+"person_test.npy").flatten()

    # Adjusting the labels so that
    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3
    y_train_valid -= 769
    y_test -= 769

    if verbose:
        print('Training/Valid data shape: {}'.format(X_train_valid.shape))
        print('Training/Valid target shape: {}'.format(y_train_valid.shape))
        print('Person train/valid shape: {}'.format(person_train_valid.shape))

        print('Test data shape: {}'.format(X_test.shape))
        print('Test target shape: {}'.format(y_test.shape))
        print('Person test shape: {}'.format(person_test.shape))

    train_valid = (X_train_valid, y_train_valid, person_train_valid)
    test = (X_test, y_test, person_test)
    return train_valid, test


def split_data(X_train_valid, y_train_valid, person_train_valid=None, prop=0.8, verbose=True):
    N = X_train_valid.shape[0]
    ind_train = np.random.choice(N, int(N*prop), replace=False)
    ind_valid = set(range(N)).difference(set(ind_train))
    ind_valid = np.array(list(ind_valid))

    X_train = X_train_valid[ind_train]
    y_train = y_train_valid[ind_train]
    person_train = person_train_valid[ind_train] if person_train_valid is not None else None

    X_valid = X_train_valid[ind_valid]
    y_valid = y_train_valid[ind_valid]
    person_valid = person_train_valid[ind_valid] if person_train_valid is not None else None

    if verbose and person_train_valid is None:
        print('Training Data:', X_train.shape, "with labels", y_train.shape)
        print('Validate Data:', X_valid.shape, "with labels", y_valid.shape)
    elif verbose:
        print('Training Data:', X_train.shape, "with labels",
              y_train.shape, "and people", person_train.shape)
        print('Validate Data:', X_valid.shape, "with labels",
              y_valid.shape, "and people", person_valid.shape)

    if person_train_valid is None:
        return (X_train, y_train), (X_valid, y_valid)
    else:
        return (X_train, y_train, person_train), (X_valid, y_valid, person_valid)


def data_prep(X, y, sub_sample=2, average=2, noise=True, noise_val=0.5):
    def noise(shape):
      if noise:
        return np.random.normal(0.0, noise_val, shape)
      else:
        return 0.0
      
    total_X = None
    total_y = None

    # Trimming the data
    X = X[:, :, 0:500]

    # Maxpooling the data
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)

    total_X = X_max
    total_y = y

    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average), axis=3)
    X_average = X_average + noise(X_average.shape)

    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))

    # Subsampling
    for i in range(sub_sample):
        X_subsample = X[:, :, i::sub_sample] + noise(X[:,:,i::sub_sample].shape)

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))

    return total_X, total_y

def load_spectrogram(X,fs=250, nperseg=32, verbose=False):
    """
    X: time domain data
    fs: sampling rate for X, default 250 for EEG dataset
    nperseg
    """
    _, _, Sxx = signal.spectrogram(X, fs, nperseg=nperseg)
    if verbose:
        print('Spectrogram Shape:', Sxx.shape)
    return Sxx