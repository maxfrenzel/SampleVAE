# Sound Sample Tool
Tool that allows for various types of new sample generation, as well as similarity search in an existing sample library.

## Making a dataset for training
Use the `make_dataset.py` script to generate a new dataset for trainig. The main parameters are `data_dir` and `dataset_name`. The former is a directory (or multiple directories) in which to look for samples (files ending in .wav, .aiff, or .mp3; only need to specify root directory, the script looks into all sub directories). The latter should a unique name for this dataset.
For example, to search for files in the directories `/Users/Shared/Decoded Forms Library/Samples` and `/Users/Shared/Maschine 2 Library`, and create the dataset called `NativeInstruments`, run the command
```
  python make_dataset.py --data_dir '/Users/Shared/Decoded Forms Library/Samples' '/Users/Shared/Maschine 2 Library' --dataset_name NativeInstruments
```
