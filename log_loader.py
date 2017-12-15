from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def plot_tensorflow_log(log_names):

    plt.figure('Validation accuracy')
    for log_name in log_names:

        validation_directory = '/home/guillaume/speech_dataset/training_logs/' + log_name + '/validation/'
        training_directory = '/home/guillaume/speech_dataset/training_logs/' + log_name + '/train/'

        files = [f for f in listdir(validation_directory) if isfile(join(validation_directory, f))]
        validation_logs_path = validation_directory + files[0]

        files = [f for f in listdir(training_directory) if isfile(join(training_directory, f))]
        training_logs_path = training_directory + files[0]

        event_acc_val = EventAccumulator(validation_logs_path)
        event_acc_tra = EventAccumulator(training_logs_path)

        event_acc_val.Reload()
        event_acc_tra.Reload()

        w_times, step_nums_acc, validation_acc = zip(*event_acc_val.Scalars('accuracy'))
        # w_times, step_nums_tra, training_acc = zip(*event_acc_tra.Scalars('accuracy'))
        print('validation of ' + log_name + ': ', validation_acc[-8:])
        plt.plot(step_nums_acc, validation_acc, '-o', label=log_name + ' validation')

    plt.xlabel("steps")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right', frameon=True)
    plt.ylim([0, 1])
    plt.show()

if __name__ == '__main__':
    models = ['lstm_100_1_1_b50', 'lstm_400_1_1_b500_long_training']
    plot_tensorflow_log(models)