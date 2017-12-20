
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argument_parser import create_parser
import sys

import numpy as np
import tensorflow as tf

import input_data
import models

FLAGS = None

def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    FLAGS.unknown_percentage = 100.0
    FLAGS.validation_percentage = 0.0
    FLAGS.testing_percentage = 0.0
    FLAGS.silence_percentage = 10.0

    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, FLAGS.num_layers, FLAGS.num_units, FLAGS.use_attn,
        FLAGS.attn_size, FLAGS)
    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    window_size_ms = str(int(FLAGS.window_size_ms))
    window_stride_ms = str(int(FLAGS.window_stride_ms))
    dct_coefficient_count = str(int(FLAGS.dct_coefficient_count))

    print('\n\npreprocessing audio files')
    print('fingerprint_size:      ', model_settings['fingerprint_size'])
    print('window_size_ms:        ', window_size_ms)
    print('window_stride_ms:      ', window_stride_ms)
    print('dct_coefficient_count: ', dct_coefficient_count)

    num_samples = audio_processor.get_size('training')
    print('num_samples: ', num_samples)
    dataset = np.empty(shape=(num_samples, model_settings['fingerprint_size']))

    audio_processor.is_idx_random = False

    print('dataset: ', dataset.shape)

    for i in range(num_samples):
      if i % 1000 == 0:
        print('status: ', i/num_samples*100.0)
      sample, _, _ = audio_processor.get_data(i, 0, model_settings, FLAGS.background_frequency, FLAGS.background_volume, time_shift_samples, 'training', sess)
      dataset[i, :] = sample.flatten()

    save_dir = '/home/guillaume/speech_dataset/train/numpy/'
    print('dataset: ', dataset.shape)
    np.save(save_dir+'train_dataset_wsize' + str(window_size_ms) + '_wstride' + window_stride_ms + '_dct' + dct_coefficient_count + '_.npy', dataset)

    # filenames = np.array([x.split('/')[-1] for x in audio_processor.testing_data])
    # np.save(save_dir+'filenames_wsize' + str(window_size_ms) + '_wstride' + window_stride_ms + '_dct' + dct_coefficient_count + '_.npy', filenames)

if __name__ == '__main__':
  parser = create_parser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
