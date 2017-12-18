import argparse


def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/guillaume/speech_dataset/',
        help="""\
         Where to download the speech training data to.
         """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
         How loud the background noise should be, between 0 and 1.
         """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
         How many of the training samples have background noise mixed in.
         """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
         How much of the training data should be silence.
         """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
         How much of the training data should be unknown words.
         """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
         Range to randomly shift the training audio by in time.
         """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=50.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='10000,10000',
        help='How many training loops to run', )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001,0.00001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        required=False,
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/home/guillaume/speech_dataset/speech_commands_train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=100,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        required=False,
        help='What model architecture to use')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        help='Number of hmm layers')
    parser.add_argument(
        '--num_units',
        type=int,
        default=100,
        help='Number of parameters in h(.;\parameters)')
    parser.add_argument(
        '--dropout_prob',
        type=float,
        default=0.5,
        help='Dropout, probability to keep units'
    )
    parser.add_argument(
        '--use_attn',
        type=bool,
        default=False,
        help='Use attention model'
    )
    parser.add_argument(
        '--attn_size',
        type=int,
        default=10,
        help='Attention model parameters'
    )
    parser.add_argument(
        '--ckpt_name',
        type=str,
        default='model',
        help='Name for saving models'
    )
    parser.add_argument(
        '--raw_data',
        type=bool,
        default=False,
        help='Use raw data'
    )
    return parser