import tensorflow as tf
import models
import input_data
from argument_parser import create_parser

import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('starting')
    parser = create_parser()
    FLAGS, unparsed = parser.parse_known_args()

    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, FLAGS.num_layers, FLAGS.num_units, False, False)

    audio_processor = input_data.AudioProcessor(
          FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
          FLAGS.unknown_percentage,
          FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
          FLAGS.testing_percentage, model_settings)
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    sess = tf.InteractiveSession()


    num_data = 10
    data, labels, names = audio_processor.get_data(num_data, 0, model_settings, FLAGS.background_frequency, FLAGS.background_volume, time_shift_samples, 'training', sess)

    print('word_to_index: ', audio_processor.word_to_index)

    spectrogram_length = model_settings['spectrogram_length']
    dct_coefficient_count = model_settings['dct_coefficient_count']

    i=0
    plt.figure()
    plt.title(names[i])
    plt.imshow(data[i, :].reshape(spectrogram_length, dct_coefficient_count)[:, :])
    plt.colorbar()
    plt.show()
    print('end')


# # (N x fingerprint_size)
# fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')