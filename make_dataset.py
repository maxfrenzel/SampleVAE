import argparse
import os
import random
import joblib


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Find all wav, aiff, and mp3 files and create dataset file.')
    parser.add_argument('--data_dir', type=str, nargs='*',
                        help='Root directory(s) in which to look for samples. Samples can be in nested directories.')
    parser.add_argument('--dataset_name', type=str,
                        help='Root directory in which to look for samples. Samples can be in nested directories.')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Percentage of (randomly chosen) files to use for training. Remaining ones are validation.')

    return parser.parse_args()


def get_data_subset(track_ids, full_data):

    categories_full = full_data['categories']
    audio_paths_full = full_data['audio_paths']
    category_names_full = full_data['category_names']

    categories = dict()
    audio_paths = dict()

    for track_id in track_ids:
        audio_paths[track_id] = audio_paths_full[track_id]

    for category_key in categories_full.keys():
        categories[category_key] = dict()
        for track_id in track_ids:
            categories[category_key][track_id] = categories_full[category_key][track_id]

    datasubset = {
        'track_ids': track_ids,
        'categories': categories,
        'audio_paths': audio_paths,
        'category_names': category_names_full
    }

    return datasubset


def main():
    args = get_arguments()

    # Get all paths of audio files
    audio_files = []

    for root_dir in args.data_dir:
        for dirName, subdirList, fileList in os.walk(root_dir, topdown=False):
            for fname in fileList:
                if os.path.splitext(fname)[1] in ['.wav', '.WAV',
                                                  '.aiff', '.AIFF',
                                                  '.aif', '.AIF',
                                                  '.mp3', '.MP3',
                                                  '.aac', '.AAC']:
                    audio_files.append('%s/%s' % (dirName, fname))

    print(f'Total number of samples found: {len(audio_files)}')

    # Build dataset
    track_ids = []
    audio_paths = dict()

    for sample_path in audio_files:
        # Find unique ID for each sample. Try filename first, if already exists add extension
        track_id = os.path.splitext(os.path.basename(sample_path))[0]
        while track_id in track_ids:
            track_id += 'x'

        audio_paths[track_id] = sample_path
        track_ids.append(track_id)

    # TODO: Add proper support for category data later; for now just pass empty dicts for compatibility
    categories = dict()
    category_names = dict()

    dataset = {
        'track_ids': track_ids,
        'categories': categories,
        'audio_paths': audio_paths,
        'category_names': category_names
    }

    # Train/valid split
    split_index = int(args.train_ratio * len(track_ids))

    # Randomize data
    random.shuffle(track_ids)

    track_ids_train = track_ids[:split_index]
    track_ids_valid = track_ids[split_index:]

    print(f'Splitting {len(track_ids)} samples into {len(track_ids_train)} training and {len(track_ids_valid)} validation samples.')

    dataset_train = get_data_subset(track_ids_train, dataset)
    dataset_valid = get_data_subset(track_ids_valid, dataset)

    print('Saving dataset files.')

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    joblib.dump(dataset_train, f'datasets/{args.dataset_name}_train.pkl')
    joblib.dump(dataset_valid, f'datasets/{args.dataset_name}_valid.pkl')

    print('Done.')


if __name__ == '__main__':
    main()
