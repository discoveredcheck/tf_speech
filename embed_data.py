
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argument_parser import create_parser
import sys

import numpy as np
import tensorflow as tf

import input_data
import models
import gc

FLAGS = None

CKPT_DIR = '/home/guillaume/speech_dataset/speech_commands_train/'

def main(_):

    import os
    os.environ['CUDA_VISIBLE_DEVICES']=""
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    FLAGS.unknown_percentage = 10.0
    FLAGS.validation_percentage = 0.0
    FLAGS.testing_percentage = 0.0
    FLAGS.silence_percentage = 10.0
    FLAGS.batch_size=5000


    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, FLAGS.num_layers, FLAGS.num_units, FLAGS.use_attn,
        FLAGS.attn_size, FLAGS)

    saver = tf.train.import_meta_graph(CKPT_DIR + FLAGS.start_checkpoint+'.meta')
    saver.restore(sess, CKPT_DIR + FLAGS.start_checkpoint)

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

    tensor_transpose = {'reconstructed': tf.get_default_graph().get_tensor_by_name('lstm_proj/BiasAdd:0'),
    'embedding': tf.get_default_graph().get_tensor_by_name('rnn/transpose:0')}

    dataset = np.empty(shape=(FLAGS.batch_size, model_settings['fingerprint_size']))


    audio_processor.is_idx_random = False

    print('dataset: ', dataset.shape)

    fingerprint_input = tf.get_default_graph().get_tensor_by_name('fingerprint_input:0')
    dropout_prob = tf.get_default_graph().get_tensor_by_name('dropout_prob:0')
    labels = []
    count = 0
    bnum = 0
    dir = './embedding_out/'
    for i in range(num_samples):
      if i % 1000 == 0:
        print('status: ', i/num_samples*100.0)

      sample, label, _ = audio_processor.get_data(i, 0, model_settings, FLAGS.background_frequency, FLAGS.background_volume, time_shift_samples, 'training', sess)
      dataset[count, :] = sample.flatten()
      labels.append(label)
      count+=1
      if (count == FLAGS.batch_size) or (i == num_samples-1):
          out_dataset = sess.run(tensor_transpose, feed_dict={fingerprint_input: dataset[0:count, :], dropout_prob: 1.0})

          [np.save(dir+k+'_'+str(bnum)+'.npy', v) for k,v in out_dataset.items()]

          np.save(dir+'labels_{}.npy'.format(bnum), np.array(labels))
          np.save(dir+'raw_{}.npy'.format(bnum), dataset[0:count, :].copy())
          print('saving {}'.format(bnum))
          count = 0
          gc.collect()
          labels = []
          bnum += 1

if __name__ == '__main__':
  parser = create_parser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
