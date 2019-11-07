import librosa
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import json

from sklearn import metrics

with open('params.json', 'r') as f:
    param = json.load(f)['audio']

N_FFT = param['N_FFT']
HOP_LENGTH = param['HOP_LENGTH']
SAMPLING_RATE = param['SAMPLING_RATE']
MELSPEC_BANDS = param['MELSPEC_BANDS']
sample_secs = None
num_samples_dataset = None


def get_params(filename):
    with open(filename, 'r') as f:
        param_full = json.load(f)

    # Check for backwards compatibility if parameters are separated or a single dict
    if 'model' in param_full.keys():
        audio_param = param_full['audio']
        model_param = param_full['model']
        param = {**param_full['audio'], **param_full['model']}
    else:
        param = param_full
        model_param = param_full
        audio_param = param_full

    return param, audio_param, model_param


# Function to read in an audio file and return a mel spectrogram
def get_melspec_old(filepath_or_audio, hop_length=HOP_LENGTH, n_mels=MELSPEC_BANDS, n_samples=num_samples_dataset,
                    sample_secs=sample_secs, as_tf_input=False):

    y_tmp = np.zeros(n_samples)

    # Load a little more than necessary as a buffer
    load_duration = None if sample_secs == None else 1.1 * sample_secs

    # Load audio file or take given input
    if type(filepath_or_audio) == str:
        y, sr = librosa.core.load(filepath_or_audio, sr=SAMPLING_RATE, mono=True, duration=load_duration)
    else:
        y = filepath_or_audio
        sr = SAMPLING_RATE

    # Truncate or pad
    if n_samples:
        if len(y) >= n_samples:
            y_tmp = y[:n_samples]
            lentgh_ratio = 1.0
        else:
            y_tmp[:len(y)] = y
            lentgh_ratio = len(y) / n_samples
    else:
        y_tmp = y
        lentgh_ratio = 1.0

    # sfft -> mel conversion
    melspec = librosa.feature.melspectrogram(y=y_tmp, sr=sr,
                                             n_fft=N_FFT, hop_length=hop_length, n_mels=n_mels)
    S = librosa.power_to_db(melspec, np.max)

    if as_tf_input:
        S = spec_to_input(S)

    return S, lentgh_ratio

def spec_to_input(spec):
    specs_out = (spec + 80.0) / 80.0
    specs_out = np.expand_dims(np.expand_dims(specs_out, axis=0), axis=3)
    return np.float32(specs_out)


def accuracy(predictions, truth, confusion=False, labels=None):

    # Turn probabilities into predictions
    class_predictions = [np.argmax(x, axis=1) for x in predictions]

    # print(class_predictions)

    accuracy = []
    precision = []
    recall = []
    f1 = []
    if confusion:
        confusion_matrix = []
        if labels is None:
            labels = len(predictions) * [None]
    
    for k in range(len(predictions)):

        # Get basic accuracy
        accuracy.append(np.sum(np.equal(class_predictions[k], truth[:,k])) / class_predictions[k].size)

        precision.append(metrics.precision_score(np.ndarray.flatten(truth[:,k]), np.ndarray.flatten(class_predictions[k]),
                                            average='micro'))
        recall.append(metrics.recall_score(np.ndarray.flatten(truth[:, k]), np.ndarray.flatten(class_predictions[k]),
                                      average='micro'))
        f1.append(metrics.f1_score(np.ndarray.flatten(truth[:, k]), np.ndarray.flatten(class_predictions[k]),
                              average='micro'))

        if confusion:
            confusion_matrix.append(metrics.confusion_matrix(np.ndarray.flatten(truth[:, k]),
                                                             np.ndarray.flatten(class_predictions[k]),
                                                             labels=labels[k]))

    if confusion:
        return accuracy, precision, recall, f1, confusion_matrix
    else:
        return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          filename=None,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    num_classes = len(classes)

    fig, ax = plt.subplots(figsize=(num_classes+3, num_classes))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label',
           ylim=[num_classes-0.5, -0.5])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
