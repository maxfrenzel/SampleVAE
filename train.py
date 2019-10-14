import tensorflow as tf
import os
from shutil import copyfile
import sys
import time
import joblib
from random import shuffle
import numpy as np
import argparse
import json

from data_reader import *
from model_iaf import *
import util

logdir = './logdir'
max_checkpoints = 5
num_steps = 10000
checkpoint_every = 500
test_every = 100
batch_size = 64
batch_size_test = 64
learning_rate = 1e-3
learning_rate_min = 1e-5
learning_rate_factor = 5.0
learning_rate_steps = 3
beta = 1.0
params = 'params.json'


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Spectrogram VAE')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='How many wav files to process at once. Default: ' + str(batch_size) + '.')
    parser.add_argument('--batch_size_test', type=int, default=batch_size_test,
                        help='Test batch size. Default: ' + str(batch_size_test) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                             'information for TensorBoard. '
                             'If the model already exists, it will restore '
                             'the state and will continue training. ')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to use for training. Only specify when initialising new model training.'
                             'If none specified, will try to use dataset in params.json file.')
    parser.add_argument('--featdir', type=str, default='./features',
                        help='Root directory in which to store the features.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=checkpoint_every,
                        help='How many steps to save each checkpoint after. Default: ' + str(checkpoint_every) + '.')
    parser.add_argument('--test_every', type=int,
                        default=test_every,
                        help='How many steps between test evaluation. Default: ' + str(test_every) + '.')
    parser.add_argument('--num_steps', type=int, default=num_steps,
                        help='Number of training steps. Default: ' + str(num_steps) + '.')
    parser.add_argument('--learning_rate', type=float, default=learning_rate,
                        help='Learning rate for training. Default: ' + str(learning_rate) + '.')
    parser.add_argument('--learning_rate_min', type=float, default=learning_rate_min,
                        help='Minimum learning rate. Stop training once reached. Default: ' + str(learning_rate_min) + '.')
    parser.add_argument('--learning_rate_factor', type=float, default=learning_rate_factor,
                        help='Factor by which to decrease learning rate when no improvement. Default: ' + str(
                            learning_rate_factor) + '.')
    parser.add_argument('--learning_rate_steps', type=int, default=learning_rate_steps,
                        help='Number of test steps without improvement to decreases learning rate. Default: ' + str(
                            learning_rate_steps) + '.')
    parser.add_argument('--beta', type=float, default=beta,
                        help='Factor for KL divergence term in loss. Default: ' + str(beta) + '.')
    parser.add_argument('--params', type=str, default=params,
                        help='JSON file with the network parameters. Default: ' + params + '.')
    parser.add_argument('--max_checkpoints', type=int, default=max_checkpoints,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(max_checkpoints) + '.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


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


def main():
    args = get_arguments()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # If restarting an existing model, look for original parameters
    if os.path.isfile(f'{args.logdir}/params.json'):
        print('Loading existing parameters.')
        print(f'{args.logdir}/params.json')
        param, audio_param, _ = get_params(f'{args.logdir}/params.json')

        if 'dropout_keep_prob' not in param.keys():
            param['dropout_keep_prob'] = 1.0
        if 'single_slice_audio' not in param.keys():
            param['single_slice_audio'] = True

    # Otherwise load new one, change the dataset, and copy to logdir
    else:
        print('Starting with new parameters.')
        # Load model parameters
        param, audio_param, model_param = get_params(args.params)

        # Adjust dataset
        audio_param['dataset_file'] = f'datasets/{args.dataset}_train.pkl'
        audio_param['dataset_test_file'] = f'datasets/{args.dataset}_valid.pkl'

        # Save parameter file
        param_dict = {"audio": audio_param,
                      "model": model_param}

        with open(f'{args.logdir}/params.json', 'w') as fp:
            json.dump(param_dict, fp, indent=2)

    # Set correct batch size in deconvolution shapes
    deconv_shape = param['deconv_shape']
    for k, s in enumerate(deconv_shape):
        actual_shape = s
        actual_shape[0] = args.batch_size
        deconv_shape[k] = actual_shape
    param['deconv_shape'] = deconv_shape

    # Create coordinator.
    coord = tf.train.Coordinator()

    with tf.name_scope('create_inputs'):
        reader = DataReader(param['dataset_file'], param, audio_param, f'{args.logdir}/params.json', coord, args.logdir,
                            featdir=args.featdir)
        spec_batch = reader.dequeue_feature(args.batch_size)
        truth_batch = reader.dequeue_truth(args.batch_size)

        batcher_test = Batcher(param['dataset_test_file'], param, audio_param, f'{args.logdir}/params.json',
                               args.logdir, featdir=args.featdir)

    num_test_data = batcher_test.num_data
    test_batches_full = int(batcher_test.num_data / args.batch_size_test)
    test_batch_last = num_test_data - (test_batches_full * args.batch_size_test)

    test_batch_features = tf.placeholder_with_default(
        input=tf.zeros([args.batch_size_test, batcher_test.num_features, batcher_test.length, 1], dtype=tf.float32),
        shape=[None, batcher_test.num_features, batcher_test.length, 1])
    test_batch_truth = tf.placeholder_with_default(
        input=tf.zeros([args.batch_size_test, batcher_test.num_categories], dtype=tf.int32),
        shape=[None, batcher_test.num_categories])
    test_batch_size_real = tf.placeholder_with_default(
        input=args.batch_size_test * tf.ones([], dtype=tf.int32),
        shape=[])

    print('Num classes: ', reader.num_classes)

    # Placeholder for dropout
    keep_prob = tf.placeholder_with_default(input=tf.to_float(1.0), shape=(), name="KeepProbRec")

    # Placeholder for learning rate and initial learning rate
    lr_placeholder = tf.placeholder_with_default(input=tf.to_float(args.learning_rate),
                                                 shape=(),
                                                 name="LearningRate")
    learning_rate = args.learning_rate

    class_labels = [[x for x in range(y)] for y in reader.num_classes]

    # Create network.
    print('Creating model.')
    net = VAEModel(param,
                   args.batch_size,
                   reader.num_categories,
                   reader.num_classes,
                   keep_prob=keep_prob)
    print('Model created.')

    print('Setting up loss.')
    loss, accuracy = net.loss(spec_batch, truth_batch, beta=args.beta)
    loss_test, accuracy_test = net.loss(test_batch_features, test_batch_truth, beta=args.beta,
                                        batch_size_real=test_batch_size_real, test=True)
    embeddings_test, prediction_test = net.embed_and_predict(test_batch_features, batch_size_real=test_batch_size_real)
    print('Loss set up.')

    optimizer = tf.train.AdamOptimizer(learning_rate=lr_placeholder,
                                       epsilon=1e-4)
    trainable = tf.trainable_variables()
    for var in trainable:
        print(var)
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(args.logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    print(summaries)

    # Set up session
    print('Setting up session.')
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    print('Session set up.')

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, args.logdir)
        if saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    print('Starting queues.')
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    print('Reader threads started.')

    step = None
    last_saved_step = saved_global_step
    mean_test_loss = 0.0
    mean_test_accuracy = batcher_test.num_categories * [0.0]
    mean_test_precision = batcher_test.num_categories * [0.0]
    mean_test_recall = batcher_test.num_categories * [0.0]
    mean_test_f1 = batcher_test.num_categories * [0.0]
    test_accuracy_string = batcher_test.num_categories * ['0.0']

    # Empty list to add previous test losses to
    test_loss_history = []

    try:
        for step in range(saved_global_step + 1, args.num_steps):
            start_time = time.time()

            epoch = reader.get_epoch(args.batch_size, step)

            # loss_value = sess.run([loss])[0]
            # print(loss_value)
            summary, loss_value, accuracy_value, _ = sess.run([summaries, loss, accuracy, optim],
                                                              feed_dict={keep_prob: param['dropout_keep_prob'],
                                                                         lr_placeholder: learning_rate})

            writer.add_summary(summary, step)

            accuracy_string = "["
            for val in accuracy_value:
                accuracy_string += '{:.3f}, '.format(val)
            accuracy_string += "]"

            if step % args.test_every == 0:
                print('Evaluating test set.')

                test_losses = []
                test_accuracies = []
                test_precisions = []
                test_recalls = []
                test_f1s = []
                confusion_matrices = []

                for step_test in tqdm(range(test_batches_full + 1)):

                    if step_test == test_batches_full:
                        test_batch_size = test_batch_last
                    else:
                        test_batch_size = args.batch_size_test

                    test_features, test_truth = batcher_test.next_batch(test_batch_size)

                    # Pad final batch
                    if step_test == test_batches_full:
                        zero_batch = np.zeros([args.batch_size_test,
                                               test_features.shape[1],
                                               test_features.shape[2],
                                               test_features.shape[3]])
                        zero_batch[:test_batch_size] = test_features

                        test_features = zero_batch

                    prediction_values, loss_value_test = sess.run([prediction_test, loss_test],
                                                                  feed_dict={test_batch_features: test_features,
                                                                             test_batch_truth: test_truth,
                                                                             test_batch_size_real: test_batch_size,
                                                                             keep_prob: 1.0})

                    test_accuracy, test_precision, test_recall, test_f1, confusion_matrix = util.accuracy(
                        prediction_values, test_truth, confusion=True, labels=class_labels)

                    test_losses.append(loss_value_test)
                    test_accuracies.append(test_accuracy)
                    test_precisions.append(test_precision)
                    test_recalls.append(test_recall)
                    test_f1s.append(test_f1)
                    confusion_matrices.append(confusion_matrix)

                mean_test_loss = np.mean(test_losses)

                # Take mean over batches
                mean_test_accuracy_array = np.mean(np.array(test_accuracies), axis=0)
                mean_test_precision_array = np.mean(np.array(test_precisions), axis=0)
                mean_test_recall_array = np.mean(np.array(test_recalls), axis=0)
                mean_test_f1_array = np.mean(np.array(test_f1s), axis=0)

                for c in range(batcher_test.num_categories):
                    mean_test_accuracy[c] = mean_test_accuracy_array[c]
                    mean_test_precision[c] = mean_test_precision_array[c]
                    mean_test_recall[c] = mean_test_recall_array[c]
                    mean_test_f1[c] = mean_test_f1_array[c]

                # Sum along batches
                # total_confusion_matrix = np.sum(np.array(confusion_matrices), axis=0)
                sum_matrices = [np.zeros_like(x) for x in confusion_matrices[0]]
                for mat_list in confusion_matrices:
                    for c, mat in enumerate(mat_list):
                        sum_matrices[c] += mat
                total_confusion_matrix = sum_matrices

                # # Split into categories
                # total_confusion_matrix = [np.squeeze(x, axis=0) for x in np.split(total_confusion_matrix,
                #                                                                   indices_or_sections=
                #                                                                   total_confusion_matrix.shape[0])]

                _summary = tf.Summary()
                _summary.value.add(tag='test/test_loss', simple_value=mean_test_loss)
                for c in range(batcher_test.num_categories):
                    _summary.value.add(tag=f'test/test_accuracy_{c}', simple_value=mean_test_accuracy[c])
                    _summary.value.add(tag=f'test/test_precision_{c}', simple_value=mean_test_precision[c])
                    _summary.value.add(tag=f'test/test_recall_{c}', simple_value=mean_test_recall[c])
                    _summary.value.add(tag=f'test/test_f1_{c}', simple_value=mean_test_f1[c])
                writer.add_summary(_summary, step)

                test_loss_history.append(mean_test_loss)

                # If no improvement over learning_rate_steps test steps, lower learning rate
                if len(test_loss_history) >= args.learning_rate_steps:
                    if test_loss_history[-args.learning_rate_steps] <= min(test_loss_history[-args.learning_rate_steps+1:]):
                        learning_rate /= args.learning_rate_factor
                        print(f'No improvement on validation data for {args.learning_rate_steps} test steps. \
                        Decreasing learning rate by factor {args.learning_rate_factor}')

                        # Check if training should be stopped
                        if learning_rate <= learning_rate_min:
                            print(f'Reached learning rate threshold of {args.learning_rate_min}. Stopping training.')
                            break

                # Plot confusion matrices
                for c in range(batcher_test.num_categories):
                    util.plot_confusion_matrix(total_confusion_matrix[c],
                                               classes=reader.class_names[c],
                                               filename=f'{args.logdir}/cm_{c}.png',
                                               normalize=True)

                test_accuracy_string = "["
                for val in mean_test_accuracy:
                    test_accuracy_string += '{:.3f}, '.format(val)
                test_accuracy_string += "]"

            duration = time.time() - start_time
            print('step {:d}; epoch {:.2f}; lr {:f} - loss = {:.3f}, accuracy = {}, test_loss = {:.3f}, test_accuracy = {}, ({:.3f} sec/step)'
                  .format(step, epoch, learning_rate, loss_value, accuracy_string, mean_test_loss, test_accuracy_string, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, args.logdir, step)
                last_saved_step = step

            _summary = tf.Summary()
            _summary.value.add(tag='epoch', simple_value=epoch)
            _summary.value.add(tag='learning_rate', simple_value=learning_rate)
            writer.add_summary(_summary, step)

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, args.logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
