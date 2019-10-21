# Sound Sample Tool
Deep learning-based tool that allows for various types of new sample generation, as well as searching for similar samples in an existing sample library.

## Making a dataset for training
Use the `make_dataset.py` script to generate a new dataset for trainig. The main parameters are `data_dir` and `dataset_name`. The former is a directory (or multiple directories) in which to look for samples (files ending in .wav, .aiff, or .mp3; only need to specify root directory, the script looks into all sub directories). The latter should a unique name for this dataset.
For example, to search for files in the directories `/Users/Shared/Decoded Forms Library/Samples` and `/Users/Shared/Maschine 2 Library`, and create the dataset called `NativeInstruments`, run the command

```
  python make_dataset.py --data_dir '/Users/Shared/Decoded Forms Library/Samples' '/Users/Shared/Maschine 2 Library' --dataset_name NativeInstruments
```

By default, the data is split randomly into 90% train and 10% validation data.

## Training a model
To train a model, use the `train.py` script. The main parameters of interest are `logdir` and `dataset`.

`logdir` specifies a unique name for the model to be trained, and creates a directory in which model checkpoints and other files are saved. Training can later be resumed from this.

`dataset` refers to a dataset created through the `make_dataset.py` script, e.g. `NativeInstruments` in the example above.

To train a model called `model_NI` on the above dataset, use

```
  python train.py --logdir model_NI --dataset NativeInstruments
```

On first training on a new dataset, all the features have to be calculated. This may take a while. When restarting the training later, or training a new model with same dataset and audio parameters, existing features are loaded.

Training automatically stops when the model converges. If no improvement is found on the validation data for several stest steps, the learning rate is lowered. Once it goes below a threshold, training stops completely.
Alternatively one can manually the training at any point.

When resuming a previously aborted model training, the dataset does not have to be specified, the script will automatically use the same dataset (and other audio and model parameters).

### Test model
A fully trained model on a very small dataset (mostly drum samples) is provided with the name `model_test`. You can use this to skip the steps of creating your dataset and training on it, and test the tool itself straight away.

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

To encode one or multiple files, pass the filenames as a list of strings to the `audio_files` parameter. If the parameter `weights` is not specified, the resulting embeddings will be averaged over before decoding into a single audio file. Alternatively, a list of numbers can be passed to `weights` to set the respective weights of each input sample in the average. Note that in the current version, the weights get normalised and negative weights might lead to undesired results or crashes. Might want to change that in the future to allow for more interesting vector arithmetic with the embeddings.

E.g. the following code combines an example kick and snare, with a 1:2 ratio:

```
tool.generate(out_file='generated.wav', audio_files=['/Users/Shared/Maschine 2 Library/Samples/Drums/Kick/Kick Accent 1.wav','/Users/Shared/Decoded Forms Library/Samples/Drums/Snare/Snare Anodyne 3.wav'], weigths=[1,2])
```

Additionally the `variance` parameter (default: 0) can be used to add some Gaussian noise before decoding, to add random variation to the samples.

### Finding similar samples
Assuming the tool was initialised with a `library_dir`, we can look for similar samples in the library.

```
tool.find_similar(target_file, num_similar=10)
```

will look for the 10 most similar samples to the sample in the audio file `target_file`. Currently the results are simply printed on screen.

### Note on sample length and audio segmentation
Currently, the tool/models treat all samples as 2 second long clips. Shorter files get padded, longer files crop.

For the purpose of building the library, an additional parameter, `library_segmentation`, can be set to `True` when initialising the tool. If `False`, files in the library are simply considered as their first 2 second. However, if `True`, the segments within longer files are considered as individual samples for the purpose of the library and similarity search.
Note that while this is implemented and technically working, the segmentation currently seems too sensitive.
