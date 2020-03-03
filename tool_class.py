import sys
import joblib
import time

# import matplotlib.pyplot as plt
# import librosa.display

from data_reader import *
from model_iaf import *
from features import *
from griffin_lim import *
from scipy.spatial import distance

# TODO: Generalise, these values shouldn't be hardcoded, maybe need to be stored in separate file when model is trained
pad_length = 125
num_features = 128


# Find n nearest neighbours in dataset to target point
# TODO: This is a dumb brute force way. Works for now, but scales terribly to large sample libraries
def n_nearest_neighbours(target, emb_dict, n=5, exclude=[]):

    distance_dict = dict()

    for t_id, emb in emb_dict.items():
        if t_id not in exclude:
            distance_dict[t_id] = distance.euclidean(target, emb)

    # Sort keys in distance dictionary by increasing value
    keys_by_value_increasing = [t[0] for t in sorted(distance_dict.items(), key=lambda x: x[1])]

    n_nn_keys = keys_by_value_increasing[:n]

    return n_nn_keys, [distance_dict[t_id] for t_id in n_nn_keys]


# Load model from checkpoint
def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


class SoundSampleTool(object):

    def __init__(self,
                 logdir,
                 batch_size=1,
                 library_dir=None,
                 library_segmentation=False):

        self.logdir = logdir
        self.batch_size = batch_size
        self.library_dir = library_dir
        self.library_segmentation = library_segmentation

        # Check if model has been trained
        if not os.path.exists(self.logdir):
            print(f'Model directory {self.logdir} not found. Train a model first.')

        # Look for original parameters
        if os.path.isfile(f'{self.logdir}/params.json'):
            print('Loading existing parameters.')
            print(f'{self.logdir}/params.json')
            self.param, self.audio_param, _ = get_params(f'{self.logdir}/params.json')
        else:
            raise ValueError('No existing parameters found. Train a model first.')

        # Check if a classifier was trained
        self.num_categories = len(self.param['predictor_units'])
        if self.num_categories > 0:
            self.has_classifier = True
            # Load class names
            self.class_names = self.param['class_names']
            self.num_classes = [len(self.class_names)]
        else:
            self.has_classifier = False
            self.num_classes = []

        # Set correct batch size in deconvolution shapes
        deconv_shape = self.param['deconv_shape']
        for k, s in enumerate(deconv_shape):
            actual_shape = s
            actual_shape[0] = self.batch_size
            deconv_shape[k] = actual_shape
        self.param['deconv_shape'] = deconv_shape

        # Create placeholders
        self.feature_placeholder = tf.placeholder_with_default(
            input=tf.zeros([self.batch_size, num_features, pad_length, 1], dtype=tf.float32),
            shape=[None, num_features, pad_length, 1])
        # self.latent_placeholder = tf.placeholder_with_default(
        #     input=tf.zeros([self.batch_size, self.param['dim_latent']], dtype=tf.float32),
        #     shape=[self.batch_size, self.param['dim_latent']])

        # Create network.
        print('Creating model.')
        self.net = VAEModel(self.param,
                            self.batch_size,
                            self.num_categories,
                            self.num_classes,
                            keep_prob=1.0)
        print('Model created.')

        # Create embedding and reconstruction tensors.
        self.embedding, self.prediction = self.net.embed_and_predict(self.feature_placeholder)
        # self.reconstruction, _ = self.net.decode(self.latent_placeholder)

        # Set up session
        print('Setting up session.')
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('Session set up.')

        # Saver for loading checkpoints of the model.
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

        try:
            self.saved_global_step = load(self.saver, self.sess, self.logdir)

        except:
            print("Something went wrong while restoring checkpoint.")
            raise

        # If a sample library directory is given, generate embeddings for all samples
        if self.library_dir is not None:
            self.sample_library = self.build_library()
        else:
            self.sample_library = None

    # Function that takes audio features and returns their embedding
    def embed(self,
              input_batch):

        # Run computation
        emb = self.sess.run([self.embedding],
                            feed_dict={self.feature_placeholder: input_batch})

        return emb

    def get_features(self,
                    input_audio,
                    offset=0.0):

        # Calculate features
        features = get_features(input_audio, param=self.param, offset=offset)
        # Add padding if too short
        if features.shape[1] < pad_length:
            features_pad = np.zeros((features.shape[0], pad_length))
            features_pad[:, :features.shape[1]] = features
            features = features_pad
        # TODO: This shouldn't be necessary at all, but sometimes get one value too many. Must be a bug in features.py..
        elif features.shape[1] > pad_length:
            features = features[:, :pad_length]
        specs_in = np.expand_dims(np.expand_dims(features, axis=0), axis=3)

        return specs_in


    # Function that takes audio sample and returns their embedding
    def embed_audio(self,
                    input_audio,
                    offset=0.0):

        specs_in = self.get_features(input_audio, offset)

        emb = self.embed(specs_in)[0]

        return emb

    # Function that takes audio sample and returns class predictions
    def predict(self,
                input_audio,
                offset=0.0):

        if self.has_classifier == False:
            print('Model has no classifier. Prediction is only available for models with classifiers.')
            return None

        input_batch = self.get_features(input_audio, offset)

        # Run computation
        probabilities = self.sess.run([self.prediction],
                                   feed_dict={self.feature_placeholder: input_batch})[0][0][0]

        p_index = np.argmax(probabilities)

        predicted_class = self.class_names[p_index]

        return probabilities, predicted_class

    # Function that takes a list of files as input, and returns combined reconstruction
    def generate(self,
                 out_file='generated.wav',
                 audio_files=[],
                 weights=[],
                 normalize_weights=True,
                 variance=0.0):

        # Process input files
        if len(audio_files) > 0:

            embeddings = []

            # Loop through files
            for k, in_file in enumerate(audio_files):

                print(f'Calculating embeddings for file {k+1} of {len(audio_files)}')

                # Embed audio
                emb = self.embed_audio(in_file)
                embeddings.append(emb)

            # If wrong number of weights is given, assume equal weighting, otherwise normalise weights
            if len(weights) != len(audio_files):
                w_list = [1.0/len(audio_files)]*len(audio_files)
            elif normalize_weights and sum(weights) == 0:
                print('Weights normalisation turned on but weights sum to zero.')
                print('Choose different weights or set normalize_weights=False.')
                print('Using uniform normalised weights instead.')
                w_list = [1.0 / len(audio_files)] * len(audio_files)
            elif normalize_weights:
                w_list = [x / sum(weights) for x in weights]
            else:
                w_list = weights

            # Weight embeddings
            embeddings = [w_list[k] * embeddings[k] for k in range(len(embeddings))]

            # Weighted mean
            embedding_mean_batch = np.sum(np.concatenate(embeddings, axis=0), axis=0, keepdims=True)

        # If no input file given, sample random point in latent space
        else:
            print('No input file given; sampling random point in latent space.')
            embedding_mean_batch = np.random.standard_normal((1, self.param['dim_latent']))

        # Add some optional Gaussian noise for variation
        if variance > 0:
            embedding_mean_batch += np.random.normal(loc=0.0,
                                                     scale=variance,
                                                     size=embedding_mean_batch.shape)

        # Decode the mean embedding
        print(f'Decoding averaged embedding.')
        out_mean = self.net.decode(np.float32(embedding_mean_batch))
        output_mean = self.sess.run(out_mean)
        spec_out = (np.squeeze(output_mean[0]) - 1.0) * 80.0

        # Reconstruct audio
        print(f'Reconstructing audio.')
        audio = griffin_lim(spec_out,
                            n_fft=self.param['N_FFT'],
                            sr=self.param['SAMPLING_RATE'],
                            mspec_bands=self.param['MELSPEC_BANDS'],
                            hop=self.param['HOP_LENGTH'])
        librosa.output.write_wav(out_file, audio / np.max(audio), sr=self.param['SAMPLING_RATE'])

    # Function that takes a sample as input, and returns closest samples in the library
    def find_similar(self,
                     target_file,
                     num_similar=1,
                     display=True):

        if self.sample_library is None:
            print('No sample library built. Specify sample directory to create library.')

            return None

        # Embed target file
        target_emb = self.embed_audio(target_file)

        # Find most similar files in library
        nn_keys, nn_distances = n_nearest_neighbours(target_emb,
                                                     self.sample_library['embeddings'],
                                                     n=num_similar,
                                                     exclude=[])

        out_text = f'{num_similar} most similar samples to {target_file}:\n'
        for k, key in enumerate(nn_keys):
            out_text += f'{k+1} - {self.sample_library["audio_paths"][key]} at {self.sample_library["onsets"][key]} (dist = {nn_distances[k]})\n'

        if display:
            print(out_text)

        # Make list of files and return
        file_list = [self.sample_library["audio_paths"][key] for key in nn_keys]
        onsets = [self.sample_library["onsets"][key] for key in nn_keys]

        return file_list, onsets, nn_distances

    def build_library(self):

        print(f'Constructing library based on directory {self.library_dir}.')

        # Get all paths of audio files
        audio_files = []

        for dirName, subdirList, fileList in os.walk(self.library_dir, topdown=False):
            for fname in fileList:
                if os.path.splitext(fname)[1] in ['.wav', '.WAV',
                                                  '.aiff', '.AIFF',
                                                  '.mp3', '.MP3']:
                    audio_files.append('%s/%s' % (dirName, fname))

        print(f'Total number of samples found in library: {len(audio_files)}')

        # Build dataset
        sample_ids = []
        audio_paths = dict()
        onsets = dict()

        for sample_path in audio_files:

            # TODO: Figure out better segmentation! Current one seems to even split e.g. cymbal sound into many segments
            # If sample segmentation activated, split longer files
            if self.library_segmentation:
                # Detect onsets with backtracking
                x, _ = librosa.load(sample_path, sr=self.param['SAMPLING_RATE'])

                # Only do if sample is longer than sample_sec
                if librosa.core.get_duration(y=x, sr=self.param['SAMPLING_RATE']) > self.param['sample_sec']:

                    onset_envelope = librosa.onset.onset_strength(x,
                                                                  sr=self.param['SAMPLING_RATE'],
                                                                  hop_length=self.param['HOP_LENGTH'])

                    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope,
                                                              sr=self.param['SAMPLING_RATE'],
                                                              hop_length=self.param['HOP_LENGTH'],
                                                              backtrack=True)

                    onset_times = librosa.frames_to_time(onset_frames,
                                                         sr=self.param['SAMPLING_RATE'],
                                                         hop_length=self.param['HOP_LENGTH'])

                    # If no onsets detected, or first onset too early in file, just use beginning of sample:
                    if len(onset_times) == 0:
                        onset_times = [0.0]
                    elif onset_times[0] < 0.5:
                        onset_times[0] = 0.0

                    # if len(onset_times) > 1:
                    #     D = np.abs(librosa.stft(x))
                    #     plt.figure()
                    #     ax1 = plt.subplot(2, 1, 1)
                    #     librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log')
                    #     times = librosa.times_like(onset_envelope, sr=self.param['SAMPLING_RATE'])
                    #     plt.title('Power spectrogram')
                    #     plt.subplot(2, 1, 2, sharex=ax1)
                    #     plt.plot(times, onset_envelope, label='Onset strength')
                    #     plt.vlines(times[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.9, linestyle='--',
                    #                label='Onsets')

                else:
                    onset_times = [0.0]
            else:
                onset_times = [0.0]

            for onset in onset_times:
                # Find unique ID for each sample. Try filename plus onset first, if already exists add extension
                sample_id = f'{os.path.splitext(os.path.basename(sample_path))[0]}_{onset}'
                while sample_id in sample_ids:
                    sample_id += 'x'

                audio_paths[sample_id] = sample_path
                onsets[sample_id] = onset
                sample_ids.append(sample_id)

        # Check if library directory exists
        library_dir = f'{self.logdir}/sample_libraries'
        if not os.path.exists(library_dir):
            print(f'Feature root directory does not yet exist. Creating {library_dir}.')
            os.makedirs(library_dir)

        library_files = [os.path.join(library_dir, fname) for fname in os.listdir(library_dir) if os.path.splitext(fname)[1] == '.pkl']

        # Check for file that contains exactly the same sample_ids
        library_dict = None
        for l_file in library_files:
            library_dict_old = joblib.load(l_file)

            if set(library_dict_old['sample_ids']) == set(sample_ids):
                library_dict = library_dict_old
                print('Found existing library file matching current library. Loading.')
                break

        # If no matching one has been found, calculate new one
        if library_dict is None:
            print('No matching existing library file found. Calculating embeddings.')

            embeddings = dict()

            # Embed each sample
            for k, s_id in enumerate(sample_ids):

                emb = self.embed_audio(audio_paths[s_id], offset=onsets[s_id])
                embeddings[s_id] = emb
                # TODO: Add support for segmentation within audio files

                if k % 100 == 0:
                    print(f'{k+1} of {len(sample_ids)} embeddings calculated.')

            library_dict = {
                'sample_ids': sample_ids,
                'audio_paths': audio_paths,
                'embeddings': embeddings,
                'onsets': onsets
            }

            # Store library file for future use
            joblib.dump(library_dict, f'{library_dir}/library_{int(time.time())}.pkl')

        return library_dict
