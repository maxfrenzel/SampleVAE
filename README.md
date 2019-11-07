# Sound Sample Tool
Deep learning-based tool that allows for various types of new sample generation, as well as sound classification, and searching for similar samples in an existing sample library. The deep learning part is implemented in TensorFlow and consists mainly of a Variational Autoencoder (VAE) with Inverse Autoregressive Flows (IAF) and an optional classifier network on top of the VAE's encoder's hidden state.

## Making a dataset for training
Use the `make_dataset.py` script to generate a new dataset for trainig. The main parameters are `data_dir` and `dataset_name`. The former is a directory (or multiple directories) in which to look for samples (files ending in .wav, .aiff, or .mp3; only need to specify root directory, the script looks into all sub directories). The latter should a unique name for this dataset.
For example, to search for files in the directories `/Users/Shared/Decoded Forms Library/Samples` and `/Users/Shared/Maschine 2 Library`, and create the dataset called `NativeInstruments`, run the command

```
  python make_dataset.py --data_dir '/Users/Shared/Decoded Forms Library/Samples' '/Users/Shared/Maschine 2 Library' --dataset_name NativeInstruments
```

By default, the data is split randomly into 90% train and 10% validation data. The optional `train_ratio` parameter (defaults to 0.9) can be used to specify a different split.

### Creating a dataset that includes class information
To add a classifier to the model, use `make_dataset_classifier.py` instead. This script works essentially in the same way as `make_dataset.py`, but it treats the immediate sub-directories in `data_dir` as class names, and assumes all samples within them belong to that resepective class.

Currently only simple multiclass classification is supported. There is also no weighting of the classes happening so you should make sure that classes are as balanced as possible. Also, the current version does the train/validation split randomly; making sure the split happens evenly across classes is a simple future improvement.

## Training a model
To train a model, use the `train.py` script. The main parameters of interest are `logdir` and `dataset`.

`logdir` specifies a unique name for the model to be trained, and creates a directory in which model checkpoints and other files are saved. Training can later be resumed from this.

`dataset` refers to a dataset created through the `make_dataset.py` script, e.g. `NativeInstruments` in the example above.

To train a model called `model_NI` on the above dataset, use

```
  python train.py --logdir model_NI --dataset NativeInstruments
```

On first training on a new dataset, all the features have to be calculated. This may take a while. When restarting the training later, or training a new model with same dataset and audio parameters, existing features are loaded.

Training automatically stops when the model converges. If no improvement is found on the validation data for several test steps, the learning rate is lowered. Once it goes below a threshold, training stops completely.
Alternatively one can manually stop the training at any point.

When resuming a previously aborted model training, the dataset does not have to be specified, the script will automatically use the same dataset (and other audio and model parameters).

If the dataset contains classification data, a confusion matrix is plotted and stored in `logdir` at every test step.

## Pre-trained Models
Three trained models are provided:

`model_general`: A model trained on slightly over 60k samples of all types. This is the same dataset that was used in my NeuralFunk project (https://towardsdatascience.com/neuralfunk-combining-deep-learning-with-sound-design-91935759d628). This model does not have a classifier.

`model_drum_classes`: A model trained on roughly 10k drum sounds, with a classifier of 9 different drum classes (e.g. kick, snare, etc).

`model_drum_machines`: A model trained on roughly 4k drum sounds, with a classifier of 71 different drum machine classes (e.g. Ace Tone Rhythm Ace, Akai XE8, Akai XR10 etc). Note that this is a tiny dataset with a very large number of classes, each only containing a handful of examples. This model is only included as an example of what's possible, not as a really useful model in itself.

## Running the sound sample tool with a trained model
To use the sample tool, start a python environment and run

```
from tool_class import *
```

You can now instantiate a SoundSampleTool class. For example to instatiate a tool based on the above model, run.

```
tool = SoundSampleTool('model_NI', library_dir='/Users/MySampleLibrary')
```

The parameter `library_dir` is optional and specifies a sample library root directory, `/Users/MySampleLibrary` in the example above. It is required to perform similarity search on this sample library. If specified, an attempt is made to load embeddings for this library. If none are found, new embeddings are calculated which may take a while (depending on sample library size).

Once completely initialised, the tool can be used for sample generation and similarity search.

Note: Currently, only one tool can be generated due to Tensorflow's graph naming conventions. To create a new tool, you have to restart your python environment.

### Generating samples
To generate new samples, use the `generate` function.

To sample a random point in latent space, decode it, and store the audio to `generated.wav`, run

```
tool.generate(out_file='generated.wav')
```

To encode one or multiple files, pass the filenames as a list of strings to the `audio_files` parameter. If the parameter `weights` is not specified, the resulting embeddings will be averaged over before decoding into a single audio file. Alternatively, a list of numbers can be passed to `weights` to set the respective weights of each input sample in the average. By default, the weights get normalised. E.g. the following code combines an example kick and snare, with a 1:2 ratio:

```
tool.generate(out_file='generated.wav', audio_files=['/Users/Shared/Maschine 2 Library/Samples/Drums/Kick/Kick Accent 1.wav','/Users/Shared/Decoded Forms Library/Samples/Drums/Snare/Snare Anodyne 3.wav'], weigths=[1,2])
```

Weight normalisation can be turned off by passing `normalize_weights=False`. This allows for arbitrary vector arithmetic with the embedding vectors, e.g. using a negative weight to subtract one vector from another.

Additionally the `variance` parameter (default: 0) can be used to add some Gaussian noise before decoding, to add random variation to the samples.

### Finding similar samples
Assuming the tool was initialised with a `library_dir`, we can look for similar samples in the library.

```
similar_files = tool.find_similar(target_file, num_similar=10)
```

will look for the 10 most similar samples to the sample in the audio file `target_file` and return them as a list, with `similar_files[0]` being the most similar, etc. By default, the results (and their respective distances in latent space) are also printed on screen.

### Classifying samples
Assuming the model was trained on a dataset with class data, we can use the tool to make class predictions for new samples.

```
probabilities, predicted_class = tool.predict(target_file)
```

will run the classifier on the provided audio file and return two items, the probability distribution over classes and the name of the most probable class.

To full list of class names (in the same order as their respective probabilities) can be accessed via `tool.class_names`.

The `offset` parameter (default: 0.0) can be used to not apply the classification to the first 2 seconds of the audio file, but a later slice of audio.

### Note on sample length and audio segmentation
Currently, the tool/models treat all samples as 2 second long clips. Shorter files get padded, longer files crop.

For the purpose of building the library, an additional parameter, `library_segmentation`, can be set to `True` when initialising the tool. If `False`, files in the library are simply considered as their first 2 second. However, if `True`, the segments within longer files are considered as individual samples for the purpose of the library and similarity search.
Note that while this is implemented and technically working, the segmentation currently seems too sensitive.
