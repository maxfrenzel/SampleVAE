import threading
import random
import tensorflow as tf
import numpy as np
import joblib
import os
import time
from tqdm import tqdm
from shutil import copyfile

from util import get_params
from features import *
# from util import *


def features_and_annotation(track_id, audio_paths, ground_truths, params):

    # TODO: If single_slice_audio, crop or pad to full audio length

    # Add some other stuff in future but for now just spec itself
    features = get_features(audio_paths, track_id=track_id, param=params)

    # Get correct annotation; go through all categories and concatenate them
    ground_truth = []
    for category_key in ground_truths:
        ground_truth.append(ground_truths[category_key][track_id])

    ground_truth = np.array(ground_truth)

    length = features.shape[1]

    return features, ground_truth, length


# Check or compute features
def generate_features(track_ids, audio_paths, ground_truths, params, audio_params, param_file, logdir,
                      feature_path_root='features', normalize=False):

    if not os.path.exists(feature_path_root):
        print(f'Feature root directory does not yet exist. Creating {feature_path_root}.')
        os.makedirs(feature_path_root)

    # Go through each directory in feature root path and check if parameters are the same
    # If a match is found, load those features. If not, generate a new directory and features.
    feature_dirs = [os.path.join(feature_path_root, name) for name in os.listdir(feature_path_root)
                    if os.path.isdir(os.path.join(feature_path_root, name))]

    directory_found = False

    for feature_dir in feature_dirs:

        # Get parameter dictionary from directory
        param_path = f'{feature_dir}/params.json'
        if os.path.exists(param_path):
            param_stored, audio_param_stored, _ = get_params(param_path)
        else:
            print(f'Paramater file missing in {feature_dir}.')
            continue

        # Compare with current parameters
        if audio_param_stored == audio_params:

            directory_found = True
            print(f'Found matching feature directory: {feature_dir}.')

            # Calculate missing feature arrays
            for k, track_id in enumerate(tqdm(track_ids)):
                if not os.path.isfile(f'{feature_dir}/{track_id}.npy'):
                    print(f'[{k}/{len(track_ids)}] Calculating missing features for {track_id}.')

                    features, ground_truth, length = features_and_annotation(track_id, audio_paths, ground_truths, params)

                    np.save(f'{feature_dir}/{track_id}.npy', features)
                    np.save(f'{feature_dir}/{track_id}_truth.npy', ground_truth)
                    np.save(f'{feature_dir}/{track_id}_length.npy', length)

            break

    # If no directory with current parameters has been found, create it and store params
    if not directory_found:

        feature_dir = os.path.join(feature_path_root, f'features_{int(time.time())}')
        print(f'Creating new feature directory: {feature_dir}.')
        os.makedirs(feature_dir)

        # Copy parameters
        print('Writing parameter file.')
        copyfile(param_file, f'{feature_dir}/params.json')

        # Calculate missing feature arrays
        for k, track_id in enumerate(tqdm(track_ids)):
            if not os.path.isfile(f'{feature_dir}/{track_id}.npy'):
                print(f'[{k}/{len(track_ids)}] Calculating missing features for {track_id}.')

                features, ground_truth, length = features_and_annotation(track_id, audio_paths, ground_truths, params)

                np.save(f'{feature_dir}/{track_id}.npy', features)
                np.save(f'{feature_dir}/{track_id}_truth.npy', ground_truth)
                np.save(f'{feature_dir}/{track_id}_length.npy', length)


    # Find normalisation factors
    norm_file = f'{logdir}/norm.pkl'
    if normalize and not os.path.isfile(norm_file):
        print('Calculating normalisation factors.')
        count = 0
        sums = []
        mins = []
        maxs = []
        for k, track_id in enumerate(tqdm(track_ids)):
            feat = np.load(f'{feature_dir}/{track_id}.npy')

            sums.append(np.sum(feat, axis=1, keepdims=True))
            mins.append(np.min(feat, axis=1, keepdims=True))
            maxs.append(np.max(feat, axis=1, keepdims=True))

            count += feat.shape[1]

        mean = np.sum(np.concatenate(sums, axis=1), axis=1, keepdims=True) / count
        max_val = np.max(np.concatenate(maxs, axis=1), axis=1, keepdims=True) - mean
        min_val = np.min(np.concatenate(mins, axis=1), axis=1, keepdims=True) - mean

        # norm = np.max(np.concatenate([max_val, np.abs(min_val)], axis=1), axis=1, keepdims=True)

        # Calculate variance
        variances = []
        for k, track_id in enumerate(tqdm(track_ids)):
            feat = np.load(f'{feature_dir}/{track_id}.npy')

            variances.append(np.sum(np.square(feat - mean), axis=1, keepdims=True))

        var = np.sum(np.concatenate(variances, axis=1), axis=1, keepdims=True) / count
        # Normalize by standard deviation
        norm = np.sqrt(var)

        norm_dict = {'mean': mean,
                     'norm': norm,
                     'min_val': min_val,
                     'max_val': max_val}

        joblib.dump(norm_dict, norm_file)

    print('Features complete.')

    return feature_dir


# Generate an index of all the time slices across all tracks
def generate_data_index(track_ids, feature_path, params, single_slice=False):

    print('Generating data index.')

    # Calculate duration of each time-window
    windows_per_sec = (params['SAMPLING_RATE'] / params['HOP_LENGTH'])

    windows_per_slice = int(params['sample_sec'] * windows_per_sec)
    overlap = 0
    if 'sample_overlap' in params.keys():
        overlap = params['sample_overlap']
    window_increment = int((params['sample_sec'] - overlap) * windows_per_sec)

    unique_slices = []

    for track_id in tqdm(track_ids):

        length = np.load(f'{feature_path}/{track_id}_length.npy')

        if single_slice:
            final_index = min(length, windows_per_slice)
            unique_slices.append((track_id, [0, final_index]))
        else:
            start_index = int(windows_per_sec * params['offset_initial'])
            end_index = int(windows_per_sec * params['offset_initial']) + windows_per_slice

            final_index = length - int(windows_per_sec * params['offset_final'])

            while end_index < final_index:
                unique_slices.append((track_id, [start_index, end_index]))
                start_index += window_increment
                end_index += window_increment

    return unique_slices


def get_input_size(track_ids, feature_path, params):

    # Calculate duration of each time-window
    windows_per_sec = (params['SAMPLING_RATE'] / params['HOP_LENGTH'])
    windows_per_slice = int(params['sample_sec'] * windows_per_sec)

    n_features = np.load(f'{feature_path}/{track_ids[0]}.npy').shape[0]

    return n_features, windows_per_slice


def randomize_data(data_list):
    for k in range(len(data_list)):
        index = random.randint(0, (len(data_list) - 1))
        yield data_list[index]


def load_norm(norm_file):
    norm_dict = joblib.load(norm_file)
    mean = norm_dict['mean']
    norm = norm_dict['norm']

    return mean, norm


def return_data(data_list, feature_path, logdir, pad_length=None, normalize=False, randomize=True):

    if randomize:
        randomized_data = randomize_data(data_list)

    # If desired, load normalisation
    if normalize:
        norm_file = f'{logdir}/norm.pkl'
        mean, norm = load_norm(norm_file)

    for data in randomized_data:

        track_id = data[0]
        indices = data[1]

        # Load features and annotations and extract correct slices
        features = np.load(f'{feature_path}/{track_id}.npy')[:, indices[0]:indices[1]]
        truth = np.array(np.load(f'{feature_path}/{track_id}_truth.npy'))
        while len(truth.shape) < 1:
            truth = np.expand_dims(truth, axis=0)

        length = features.shape[1]

        # Pad to uniform length
        if pad_length is not None and length < pad_length:
            features = np.pad(features, ((0, 0), (0, pad_length - length)), 'constant', constant_values=((0, 0), (0, 0)))
        # elif length > pad_length:
        #     # This shouldn't happen given the preprocessing used here
        #     raise Exception(f'Feature too long: {features.shape[1]}: {track_id}!')

        # Normalise
        if normalize:
            features -= mean
            features /= norm

        # Save actual length
        length = np.expand_dims(length, axis=0)

        # Add channel dimension to features
        features = np.expand_dims(features, axis=2)

        yield features, truth, length


class DataReader(object):
    def __init__(self,
                 dataset_file,
                 params,
                 audio_params,
                 param_file,
                 coord,
                 logdir,
                 featdir=None,
                 queue_size=128):

        self.params = params
        self.audio_params = audio_params
        self.param_file = param_file
        self.ground_truth, self.audio_paths, self.track_ids, self.num_categories, self.num_classes, self.class_names = load_dataset_file(
            dataset_file)
        self.coord = coord
        self.logdir = logdir
        self.featdir = featdir
        self.threads = []

        # Make sure number of categories are compatible with parameters
        assert len(params['predictor_units']) == self.num_categories, "Number of categories in data does not match " \
                                                                      "parameter file. Update units for predictor in " \
                                                                      "params."

        # Check if features are calculated already or still need to be generated
        self.feature_dir = generate_features(self.track_ids,
                                             self.audio_paths,
                                             self.ground_truth,
                                             self.params,
                                             self.audio_params,
                                             self.param_file,
                                             logdir=self.logdir,
                                             feature_path_root=self.featdir,
                                             normalize=params['feature_normalization'])

        self.data_index = generate_data_index(self.track_ids,
                                              self.feature_dir,
                                              self.params,
                                              single_slice=self.params['single_slice_audio'])

        self.num_data = len(self.data_index)
        print('Total amount of data: ', self.num_data)

        self.num_features, self.length = get_input_size(self.track_ids, self.feature_dir, self.params)

        print("Feature length: ", self.length)

        # TODO: Dimensions shouldn't be hardcoded here.
        self.feature_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.feature_queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[((128, self.length, 1))])
        self.feature_enqueue = self.feature_queue.enqueue([self.feature_placeholder])

        self.truth_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.truth_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                            shapes=[(self.num_categories, )])
        self.truth_enqueue = self.truth_queue.enqueue([self.truth_placeholder])

        # self.length_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        # self.length_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
        #                                        shapes=[(1,)])
        # self.length_enqueue = self.length_queue.enqueue([self.length_placeholder])

    def dequeue_feature(self, num_elements):
        output = self.feature_queue.dequeue_many(num_elements)
        return output

    def dequeue_truth(self, num_elements):
        output = self.truth_queue.dequeue_many(num_elements)
        return output

    # def dequeue_length(self, num_elements):
    #     output = self.length_queue.dequeue_many(num_elements)
    #     return output

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = return_data(self.data_index, self.feature_dir,
                                   logdir=self.logdir,
                                   pad_length=self.length,
                                   normalize=self.params['feature_normalization'])
            count = 0
            for feature, truth, length in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                sess.run(self.feature_enqueue,
                         feed_dict={self.feature_placeholder: feature})
                sess.run(self.truth_enqueue,
                         feed_dict={self.truth_placeholder: truth})
                # sess.run(self.length_enqueue,
                #          feed_dict={self.length_placeholder: length})

                count += 1

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

    def get_epoch(self, batch_size, step):
        return (batch_size * step) / self.num_data


class Batcher(object):
    def __init__(self,
                 dataset_file,
                 params,
                 audio_params,
                 param_file,
                 logdir,
                 featdir=None,
                 shuffle=False):

        self.params = params
        self.audio_params = audio_params
        self.param_file = param_file
        self.ground_truth, self.audio_paths, self.track_ids, self.num_categories, self.num_classes, self.class_names = load_dataset_file(
            dataset_file)
        self.logdir = logdir
        self.featdir = featdir
        self.shuffle = shuffle

        # Make sure number of categories are compatible with parameters
        assert len(params['predictor_units']) == self.num_categories, "Number of categories in data does not match " \
                                                                      "parameter file. Update units for predictor in " \
                                                                      "params."

        # Check if features are calculated already or still need to be generated
        self.feature_dir = generate_features(self.track_ids,
                                             self.audio_paths,
                                             self.ground_truth,
                                             self.params,
                                             self.audio_params,
                                             self.param_file,
                                             logdir=self.logdir,
                                             feature_path_root=self.featdir,
                                             normalize=params['feature_normalization'])

        self.data_index = generate_data_index(self.track_ids,
                                              self.feature_dir,
                                              self.params,
                                              single_slice=self.params['single_slice_audio'])

        if self.shuffle:
            np.random.shuffle(self.data_index)

        self.num_data = len(self.data_index)
        print('Total amount of data: ', self.num_data)

        self.num_features, self.length = get_input_size(self.track_ids, self.feature_dir, self.params)

        self.index = 0

        if self.params['feature_normalization']:
            self.mean, self.norm = load_norm(f'{self.logdir}/norm.pkl')

    def get_epoch(self, batch_size, step):
        return (batch_size * step) / self.num_data

    def next_batch(self, batch_size):

        feature_list = []
        truth_list = []

        data_iterator = return_data(self.data_index, self.feature_dir,
                                    logdir=self.logdir,
                                    pad_length=self.length,
                                    normalize=self.params['feature_normalization'])

        for k in range(batch_size):

            # Return features from generator, possibly recreating it if it's empty
            try:
                features, truth, length = next(data_iterator)
            except:
                # Recreate the generator
                data_iterator = return_data(self.data_index, self.feature_dir,
                                            logdir=self.logdir,
                                            pad_length=self.length,
                                            normalize=self.params['feature_normalization'])
                features, truth, length = next(data_iterator)

            feature_list.append(np.float32(np.expand_dims(features, axis=0)))
            truth_list.append(np.expand_dims(truth, axis=0))

            self.index += 1
            if self.index == self.num_data:
                self.index = 0

                if self.shuffle:
                    np.random.shuffle(self.data_index)

        feature_batch = np.concatenate(feature_list, axis=0)
        truth_batch = np.concatenate(truth_list, axis=0)

        return feature_batch, truth_batch


def load_dataset_file(filename='dataset.pkl'):

    print('Loading dataset.')

    dataset = joblib.load(filename)

    ground_truth = dataset['categories']
    audio_paths = dataset['audio_paths']
    track_ids = dataset['track_ids']

    # Find number of categories and number of classes in each category
    category_count = 0
    class_count = []
    class_names = []
    for category_key in dataset['categories']:

        class_set = set()
        for class_key in dataset['categories'][category_key]:
            class_set.add(dataset['categories'][category_key][class_key])

        category_count += 1
        class_count.append(len(dataset['category_names'][category_key]))

        class_names.append(dataset['category_names'][category_key])

    return ground_truth, audio_paths, track_ids, category_count, class_count, class_names
