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

import input_data_prediction
import models
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    test_fingerprints = np.load('../speech_dataset/test/numpy/test_dataset_wsize50_wstride10_dct40_.npy')
    filenames = np.load('../speech_dataset/test/numpy/filenames_wsize50_wstride10_dct40_.npy')

    assert len(test_fingerprints) == len(filenames)

    print('test_fingerprints: ', test_fingerprints.shape)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    model_settings = models.prepare_model_settings(
      len(input_data_prediction.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, FLAGS.num_layers, FLAGS.num_units, FLAGS.use_attn, FLAGS.attn_size, FLAGS)

    fingerprint_size = test_fingerprints.shape[1]

    # (N x fingerprint_size)
    fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

    # Define loss and optimizer
    predicted_indices = tf.argmax(logits, 1)

    print('\n\nFLAGS ===>', FLAGS)

    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

    names = dict()
    names[0] = 'silence'
    names[1] = 'unknown'
    names[2] = 'yes'
    names[3] = 'no'
    names[4] = 'up'
    names[5] = 'down'
    names[6] = 'left'
    names[7] = 'right'
    names[8] = 'on'
    names[9] = 'off'
    names[10] = 'stop'
    names[11] = 'go'

    with open('predictions.txt', 'w') as f:
        f.write('fname,label\n')
        bsize = FLAGS.batch_size
        for i in range(0, len(test_fingerprints), bsize):
            if i % 10000 == 0:
                print('batch: '+str(i))
            en = min(i+bsize, len(test_fingerprints))
            predictions = sess.run(predicted_indices,
            feed_dict={
                fingerprint_input: test_fingerprints[i:en, :],
                dropout_prob: 1.0
            })
            for a, b in zip(filenames[i:en], predictions):
                f.write(a+','+names[b]+'\n')


if __name__ == '__main__':
  parser = create_parser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
