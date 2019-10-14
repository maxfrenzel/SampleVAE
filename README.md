# Sound Sample Tool
Tool that allows for various types of new sample generation, as well as similarity search in an existing sample library.

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
