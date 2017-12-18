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
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

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

    # (N x fingerprint_size)
    fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(tf.float32, [None, label_count], name='groundtruth_input')

    # Define loss and optimizer
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    print('\n\nFLAGS ===>', FLAGS)

    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

    print('finished loading')
    set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', set_size)

    set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    prob_list = []
    predict_idx_list = []
    expeted_idx_list = []
    emb_list = []
    label_list = []
    for i in xrange(0, set_size, FLAGS.batch_size):
        test_fingerprints, test_ground_truth, _ = audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
        if True:
            test_accuracy, conf_matrix, prob, predict_idx, expected_idx = sess.run(
                [evaluation_step, confusion_matrix, tf.nn.softmax(logits), predicted_indices, expected_indices],
                feed_dict={
                    fingerprint_input: test_fingerprints,
                    ground_truth_input: test_ground_truth,
                    dropout_prob: 1.0
                })

            prob_list.append(prob)
            predict_idx_list.append(predict_idx)
            expeted_idx_list.append(expected_idx)

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (test_accuracy * batch_size) / set_size
            if total_conf_matrix is None:
              total_conf_matrix = conf_matrix
            else:
              total_conf_matrix += conf_matrix
        else:
            emb = sess.run(tf.get_default_graph().get_tensor_by_name('rnn/transpose:0')[:, -1, :],
                           feed_dict={fingerprint_input: test_fingerprints,
                                      ground_truth_input: test_ground_truth,
                                      dropout_prob: 1.0})
            emb_list.append(emb)
            label_list.append(test_ground_truth)

    if True:
        prob_numpy = np.concatenate(prob_list)
        predict_idx_numpy = np.concatenate(predict_idx_list)
        expeted_idx_numpy = np.concatenate(expeted_idx_list)
        print(prob_numpy.shape, predict_idx_numpy.shape, expeted_idx_numpy.shape)
    else:
        np.save('embedding_test.npy', np.concatenate(emb_list))
        np.save('labels_test.npy', np.concatenate(label_list))



    np.save('prob_numpy.npy', prob_numpy)
    np.save('predict_idx_numpy.npy', predict_idx_numpy)
    np.save('expeted_idx_numpy.npy', expeted_idx_numpy)

    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    for _c in range(label_count):
        tf.logging.info('Testing without %d = %.1f%%' % (_c, 100 * (total_conf_matrix.trace() - total_conf_matrix[_c,_c]) / (total_conf_matrix.sum().sum() - total_conf_matrix[_c,:].sum() - total_conf_matrix[:, _c].sum() +total_conf_matrix[_c,_c])))

    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100, set_size))




if __name__ == '__main__':
  parser = create_parser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
