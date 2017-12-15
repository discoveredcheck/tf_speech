import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

SUMMARY_DIR = '/home/ashukla/devel/spch/speech_commands/saved_models'


def get_summaries_dir(name):
    return '--summaries_dir=' + SUMMARY_DIR + name


def get_model_architecture(name):
    return '--model_architecture=' + name


def get_dropout_prob(prob):
    return '--dropout_prob=' + str(prob)


def get_num_units(num_units):
    return '--num_units=' + str(num_units)


def get_num_layers(num_layers):
    return '--num_layers=' + str(num_layers)

def get_batch(num):
    return '--batch_size=' + str(num)

def get_use_atten(is_use):
    if is_use:
        return '--use_attn=1'
    else:
        return '--use_attn=0'

def get_attn_param(num_param):
    return '--attn_size=' + str(num_param)

def run(model_parameters):
    print(bcolors.OKGREEN + bcolors.BOLD + 'run: ', " ".join(model_parameters) + bcolors.ENDC)
    os.system('python train.py ' + " ".join(model_parameters))

def run_test(model_parameters):
    print(bcolors.OKGREEN + bcolors.BOLD + 'test run: ', " ".join(model_parameters) + bcolors.ENDC)
    os.system('python test_model.py ' + " ".join(model_parameters))

def run_prediction(model_parameters):
    print(bcolors.OKGREEN + bcolors.BOLD + 'test run: ', " ".join(model_parameters) + " --data_dir=/home/ashukla/devel/spch/speech_commands/saved_models" + bcolors.ENDC)
    os.system('python test_model.py ' + " ".join(model_parameters))


if __name__ == '__main__':
    run_mode = 'prediction'

    if run_mode == 'prediction':
        run_prediction([get_model_architecture('lstm'), get_num_units(1200), get_num_layers(1), get_dropout_prob(0.9), get_batch(200), '--start_checkpoint=/home/ashukla/devel/spch/speech_commands/saved_models/lstm.ckpt-8200'])
    elif run_mode == 'test':
        run_test([get_model_architecture('lstm'), get_num_units(1200), get_num_layers(1), get_dropout_prob(0.9), get_batch(200), '--start_checkpoint=/home/ashukla/devel/spch/speech_commands/saved_models/lstm.ckpt-8200'])
    else:
        models = []
        models.append([get_model_architecture('lstm'), get_num_units(1200), get_num_layers(1), get_dropout_prob(0.9), get_batch(200), get_summaries_dir('lstm_12000_drop_0.9_sw100_batch200')])

        #models.append([get_model_architecture('lstm'), get_num_units(400), get_num_layers(1), get_dropout_prob(1.0), get_batch(10), get_summaries_dir('lstm_400_1_1_b10')])
        #models.append([get_model_architecture('lstm'), get_num_units(400), get_num_layers(1), get_dropout_prob(1.0), get_batch(25), get_summaries_dir('lstm_400_1_1_b25')])
        #models.append([get_model_architecture('lstm'), get_num_units(400), get_num_layers(1), get_dropout_prob(1.0), get_batch(50), get_summaries_dir('lstm_400_1_1_b50')])
        #models.append([get_model_architecture('lstm'), get_num_units(400), get_num_layers(1), get_dropout_prob(1.0), get_batch(100), get_summaries_dir('lstm_400_1_1_b100')])
        #models.append([get_model_architecture('lstm'), get_num_units(400), get_num_layers(1), get_dropout_prob(1.0), get_batch(500), get_summaries_dir('lstm_400_1_1_b500_long_training')])

        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(1.0), get_batch(10), get_summaries_dir('lstm_100_1_1_b10')])
        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(1.0), get_batch(25), get_summaries_dir('lstm_100_1_1_b25')])
        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(1.0), get_batch(50), get_summaries_dir('lstm_100_1_1_b50')])
        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(1.0), get_batch(100), get_summaries_dir('lstm_100_1_1_b100')])
        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(1.0), get_batch(500), get_summaries_dir('lstm_100_1_1_b500')])



        #models.append([get_model_architecture('lstm'), get_num_units(50), get_num_layers(1), get_dropout_prob(1.0), get_summaries_dir('lstm_50_1_1')])
        #models.append([get_model_architecture('lstm'), get_num_units(50), get_num_layers(1), get_dropout_prob(0.9), get_summaries_dir('lstm_50_1_0.9')])
        # models.append([get_model_architecture('lstm'), get_num_units(50), get_num_layers(1), get_dropout_prob(0.7), get_summaries_dir('lstm_50_1_0.7')])
        # models.append([get_model_architecture('lstm'), get_num_units(50), get_num_layers(1), get_dropout_prob(0.5), get_summaries_dir('lstm_50_1_0.5')])

        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(1.0), get_summaries_dir('lstm_100_1_1')])
        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(0.9), get_summaries_dir('lstm_100_1_0.9')])
        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(0.7), get_summaries_dir('lstm_100_1_0.7')])
        #models.append([get_model_architecture('lstm'), get_num_units(100), get_num_layers(1), get_dropout_prob(0.5), get_summaries_dir('lstm_100_1_0.5')])

        #models.append([get_model_architecture('lstm'), get_num_units(400), get_num_layers(4), get_dropout_prob(0.9), get_summaries_dir('lstm_400_4_0.9')])
        #models.append([get_model_architecture('lstm'), get_num_units(200), get_num_layers(1), get_dropout_prob(0.9), get_summaries_dir('lstm_200_1_0.9')])

        #models.append([get_model_architecture('lstm'), get_num_units(500), get_num_layers(1), get_dropout_prob(1.0), get_summaries_dir('lstm_500_1_1')])
        #models.append([get_model_architecture('lstm'), get_num_units(200), get_num_layers(1), get_dropout_prob(0.9), get_summaries_dir('lstm_500_1_0.9')])

        #models.append([get_model_architecture('lstm'), get_num_units(200), get_num_layers(1), get_dropout_prob(0.7), get_summaries_dir('lstm_200_1_0.7')])
        #models.append([get_model_architecture('lstm'), get_num_units(200), get_num_layers(1), get_dropout_prob(0.5), get_summaries_dir('lstm_200_1_0.5')])


        i=1
        for model in models:
            print(bcolors.OKBLUE + bcolors.BOLD + 'model : ' + str(i) + '/' + str(len(models)) + bcolors.ENDC)
            run(model)
            i+=1
