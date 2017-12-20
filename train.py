# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argument_parser import create_parser
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  print('\nFLAGS====> ', FLAGS)

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, FLAGS.num_layers, FLAGS.num_units, FLAGS.use_attn, FLAGS.attn_size, FLAGS)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  # (N x fingerprint_size)
  fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')

  if FLAGS.pretrain == 1:
    output, state, dropout_prob = models.create_model(
          fingerprint_input,
          model_settings,
          FLAGS.model_architecture,
          is_training=True)
  else:
    logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

    # Define loss and optimizer
  ground_truth_input = tf.placeholder(tf.float32, [None, label_count], name='groundtruth_input')
    
  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  if FLAGS.pretrain==1:
    # Create the unsupervised pretraining loss
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_2d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])

    cell = tf.contrib.rnn.LSTMCell(model_settings['num_units'], use_peepholes=True)
    init_state = tf.concat((state[0].c, state[0].h), -1)
    dec_state = state[0]
    dec_input_ = tf.zeros(shape=(tf.shape(fingerprint_2d)[0], model_settings['num_units']), dtype=tf.float32)
    dec_outputs = []
    for step in range(input_time_size):
        dec_input_, dec_state = cell(tf.concat((dec_input_, init_state), -1), dec_state)
        dec_outputs.append(dec_input_)
    dec_outputs = tf.stack(dec_outputs, axis=1)
    projected = tf.layers.dense(dec_outputs, units=input_frequency_size, use_bias=True, name='lstm_proj')

    next_word = tf.pad(fingerprint_2d[:,1:,:], [[0,0],[0,1],[0,0]])

    cross_entropy_mean = tf.reduce_mean(tf.square(next_word - projected))
  else:
    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('cross_entropy'):
      cross_entropy_mean = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              labels=ground_truth_input, logits=logits))

  tf.summary.scalar('cross_entropy', cross_entropy_mean)
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy_mean)

  if FLAGS.pretrain==1:
    evaluation_step = cross_entropy_mean
    confusion_matrix = tf.no_op()
    predicted_indices = tf.no_op()
    expected_indices = tf.no_op()
    correct_prediction = tf.no_op()
  else:
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  tf.summary.scalar('accuracy', evaluation_step)
  global_step = tf.contrib.framework.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_step=0
  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir, FLAGS.ckpt_name + '.pbtxt')

  # Save list of words.
  with gfile.GFile(os.path.join(FLAGS.train_dir, FLAGS.ckpt_name + '_labels.txt'), 'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  # Training loop.
  training_steps_max = np.sum(training_steps_list)
  emb_list = []
  label_list=[]
  for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
    # Pull the audio samples we'll use for training.
    # FLAGS.batch_size
    train_fingerprints, train_ground_truth, _ = audio_processor.get_data(FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency, FLAGS.background_volume, time_shift_samples, 'training', sess)
    # Run the graph with this batch of training data.
    if True:
      train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
              [
                  merged_summaries, evaluation_step, cross_entropy_mean, train_step,
                  increment_global_step
              ],
              feed_dict={
                  fingerprint_input: train_fingerprints,
                  ground_truth_input: train_ground_truth,
                  learning_rate_input: learning_rate_value,
                  dropout_prob: FLAGS.dropout_prob
              })
      train_writer.add_summary(train_summary, training_step)
      if training_step % 1000 == 0:
        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' % (
          training_step, learning_rate_value, train_accuracy * 100, cross_entropy_value))
    else:
        emb = sess.run(tf.get_default_graph().get_tensor_by_name('rnn/transpose:0')[:,-1,:], feed_dict={fingerprint_input: train_fingerprints,
                                                               ground_truth_input: train_ground_truth,
                                                                learning_rate_input: learning_rate_value,
                                                                dropout_prob: FLAGS.dropout_prob})
        emb_list.append(emb)
        label_list.append(train_ground_truth)



    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth, _ = (audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run(
              [merged_summaries, evaluation_step, confusion_matrix],
              feed_dict={
                  fingerprint_input: validation_fingerprints,
                  ground_truth_input: validation_ground_truth,
                  dropout_prob: 1.0
              })
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix

      if FLAGS.pretrain==1:
        tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' % (training_step, total_accuracy * 100, set_size))
      else:
        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Step %d: Validation without Unk = %.1f%%' % (training_step, 100*(total_conf_matrix.trace() - total_conf_matrix[1, 1]) / (total_conf_matrix.sum().sum() - total_conf_matrix[1, :].sum() - total_conf_matrix[:, 1].sum() + total_conf_matrix[1, 1])))
        tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' % (training_step, total_accuracy * 100, set_size))


    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.ckpt_name + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)

  if False:
    np.save('embedding.npy', np.concatenate(emb_list))
    np.save('labels.npy', np.concatenate(label_list))
    exit(1)
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100, set_size))


if __name__ == '__main__':
  parser = create_parser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
