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
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count, num_layers, num_units, use_attn, attn_size, flags):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.
    num_layers: Number of HMMs.
    num_units: Number of parmeters for each state transition function h(.)

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count*spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'num_layers': num_layers,
      'num_units': num_units,
      'use_attn' : use_attn,
      'attn_size': attn_size,
      'raw_data': flags.raw_data
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'conv1d':
    return create_conv1d_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,  is_training)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings, is_training, runtime_settings)
  elif model_architecture == 'attn':
    return create_att_lstm_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'lstm':
    return create_lstm_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'pretrain_attn':
      return create_pretrain_attn(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'linear':
    return create_linear_model(fingerprint_input, model_settings, is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):
  """Builds a model with a single hidden fully-connected layer.

  This is a very simple model with just one matmul and bias layer. As you'd
  expect, it doesn't produce very accurate results, but it is very fast and
  simple, so it's useful for sanity testing.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.Variable(
      tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(fingerprint_input, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_pretrain_attn(fingerprint_input, model_settings, is_training):

    model_settings['pretrain'] = True
    if is_training:
        output, dropout = create_lstm_model(fingerprint_input, model_settings, is_training)
    else:
        output = create_lstm_model(fingerprint_input, model_settings, is_training)


def create_att_lstm_model(fingerprint_input, model_settings, is_training):
    """
    Vanila lstm
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    number_of_layers = model_settings['num_layers']
    lstm_size = model_settings['num_units']
    use_attn = model_settings['use_attn']
    attn_size = 100
    attn_type = 'softmax'

    fingerprint_2d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])

    def lstm_cell(lstm_size): return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(lstm_size, use_peepholes=True), output_keep_prob=dropout_prob)

    def bilstm_cell(lstm_size):
        return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(lstm_size / 2), output_keep_prob=dropout_prob)


    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size) for _ in range(number_of_layers)])
    output, state = tf.nn.dynamic_rnn(stacked_lstm, fingerprint_2d, dtype=tf.float32)

    #stacked_lstm_fw = bilstm_cell(lstm_size)
    #stacked_lstm_bw = bilstm_cell(lstm_size)
    #output, state = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw, fingerprint_2d, dtype=tf.float32)
    #output = tf.concat(output, 2)

    label_count = model_settings['label_count']

    print('Using attention')
    memory = tf.layers.dense(output, units=attn_size, activation=tf.tanh, use_bias=True)
    query = tf.get_variable(shape=(attn_size,), dtype=tf.float32, name='attn_query')
    bias = tf.get_variable(shape=(attn_size,), dtype=tf.float32, name='attn_bias')
    p = tf.reduce_sum((memory * query) + bias, axis=-1)
    if attn_type == 'softmax':
        weights = tf.nn.softmax(p, dim=-1)
    elif attn_type == 'relu':
        weights = tf.nn.relu(p)
    elif attn_type == 'tanh':
        weights = tf.tanh(p)

    # weights = tf.Print(weights, [weights], 'Weights')

    final_fc = tf.reduce_sum(output * tf.expand_dims(weights, -1), axis=1)
    final_fc = tf.layers.dense(final_fc, label_count)

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_lstm_model(fingerprint_input, model_settings, is_training):
    """
    Vanila lstm
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    number_of_layers = model_settings['num_layers']
    lstm_size = model_settings['num_units']
    is_pretraining = 'pretrain' in model_settings and model_settings['pretrain']

    fingerprint_2d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])
    fingerprint_2d = fingerprint_2d[:, :, :]

    #mask = tf.get_variable(shape=(input_frequency_size), dtype=tf.float32, name='freq_mask')
    #fingerprint_2d = fingerprint_2d * mask
    def lstm_cell(lstm_size): return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(lstm_size, use_peepholes=True), output_keep_prob=dropout_prob)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size) for _ in range(number_of_layers)])
    output, state = tf.nn.dynamic_rnn(stacked_lstm, fingerprint_2d, dtype=tf.float32)

    if is_pretraining:
        if is_training:
            return output, dropout_prob
        else:
            return output

    label_count = model_settings['label_count']

    final_fc = tf.layers.dense(output[:, -1, :], label_count)

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_linear_model(fingerprint_input, model_settings, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']

    fingerprint_1d = tf.reshape(fingerprint_input, [-1, input_frequency_size*input_time_size])

    W = tf.Variable(tf.truncated_normal([input_frequency_size*input_time_size, label_count], stddev=0.01))
    b = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(fingerprint_1d, W) + b
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(second_conv_output_width * second_conv_output_height * second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(tf.truncated_normal([second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a convolutional model with low compute requirements.

  This is roughly the network labeled as 'cnn-one-fstride4' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces slightly lower quality results than the 'conv' model, but needs
  fewer weight parameters and computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 1
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x)
  first_conv_output_height = math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y)
  first_conv_element_count = int(
      first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,
                                    [-1, first_conv_element_count])
  first_fc_output_channels = 128
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_conv_element_count, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 128
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_conv1d_model(fingerprint_input, model_settings, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_3d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])
  fingerprint_3d = fingerprint_3d[:, :, 1:]

  first_filter_width = 8
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_width, input_frequency_size-1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv1d(fingerprint_3d, first_weights, 1, 'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.squeeze(tf.nn.max_pool(tf.expand_dims(first_dropout, -1), [1, 2, 1, 1], [1, 2, 1, 1], 'SAME'), axis=-1)
  second_filter_width = 4
  second_filter_count = 64
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv1d(max_pool, second_weights, 1,
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  conv_out_dim = second_conv_shape[2]
  conv_out_time = second_conv_shape[1]
  second_conv_element_count = int(conv_out_time * conv_out_dim)
  flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(tf.truncated_normal([second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_autoencoder_conv1d_model(fingerprint_input, model_settings, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_frequency_size    = model_settings['dct_coefficient_count']
    input_time_size         = model_settings['spectrogram_length']
    fingerprint_3d          = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])
    fingerprint_3d          = fingerprint_3d[:, :, 1:]

    first_filter_width = 8
    first_filter_count = 64
    first_weights = tf.Variable(tf.truncated_normal([first_filter_width, input_frequency_size - 1, first_filter_count], stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv1d(fingerprint_3d, first_weights, 1, 'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    tf.nn
    max_pool = tf.squeeze(tf.nn.max_pool(tf.expand_dims(first_dropout, -1), [1, 2, 1, 1], [1, 2, 1, 1], 'SAME'), axis=-1)

    second_filter_width = 4
    second_filter_count = 64
    second_weights = tf.Variable(tf.truncated_normal([second_filter_width, first_filter_count, second_filter_count ], stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv1d(max_pool, second_weights, 1, 'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu
    second_conv_shape = second_dropout.get_shape()
    conv_out_dim = second_conv_shape[2]
    conv_out_time = second_conv_shape[1]

    second_conv_element_count = int(conv_out_time * conv_out_dim)

    flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc