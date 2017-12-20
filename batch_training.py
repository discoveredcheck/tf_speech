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

#SUMMARY_DIR = '/home/ashukla/devel/spch/speech_dataset/training_logs/'
SUMMARY_DIR = '/home/guillaume/speech_dataset/training_logs/'

def get_summaries_dir_ckpt(name):
    return '--summaries_dir=' + SUMMARY_DIR + name + ' --ckpt_name=' + name


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


def use_raw(value):
    if value:
        return '--raw_data=1'
    else:
        return '--raw_data=0'

def get_pretrain(value):
    if value:
        return '--pretrain=1'
    else:
        return '--pretrain=0'


def set_unknown(value):
    return '--unknown_percentage='+str(value)


def get_attn_param(num_param):
    return '--attn_size=' + str(num_param)

def get_check_point(name):
    return '--start_checkpoint='+ckpt_dir+name

def run(model_parameters):
    print(bcolors.OKGREEN + bcolors.BOLD + 'run: ', " ".join(model_parameters) + bcolors.ENDC)
    os.system('python train.py ' + " ".join(model_parameters))

def run_test(model_parameters):
    print(bcolors.OKGREEN + bcolors.BOLD + 'test run: ', " ".join(model_parameters) + bcolors.ENDC)
    os.system('python test.py ' + " ".join(model_parameters))

def run_prediction(model_parameters):
    print(bcolors.OKGREEN + bcolors.BOLD + 'test run: ', " ".join(model_parameters) + bcolors.ENDC)
    os.system('python predict.py ' + " ".join(model_parameters))


if __name__ == '__main__':
    import sys
    run_mode = sys.argv[1]

    ckpt = 'lstm_800_drop_0.7_unknown90.ckpt-18000'
    units = 800
    model_name = 'lstm'

    ckpt_dir = '/home/guillaume/speech_dataset/speech_commands_train/'
    if run_mode == 'predict':
        run_prediction([get_model_architecture(model_name), get_num_units(units), get_num_layers(1), get_batch(100), '--start_checkpoint=' + ckpt_dir + ckpt])
    elif run_mode == 'test':
        run_test([get_model_architecture(model_name), get_num_units(units), get_num_layers(1), get_batch(100), '--start_checkpoint=' + ckpt_dir + ckpt])
    else:
        models = []
        # models.append([get_model_architecture('lstm'), get_num_units(600), get_num_layers(1), get_dropout_prob(0.8), set_unknown(50.0), get_batch(100), get_pretrain(False), get_summaries_dir_ckpt('lstm_600_drop_0.8_unknown50'), get_check_point(ckpt)])
        models.append([get_model_architecture('lstm'), get_num_units(800), get_num_layers(1), get_dropout_prob(0.7), set_unknown(90.0), get_batch(100), get_pretrain(True), get_summaries_dir_ckpt('lstm_800_drop_0.7_unknown90_semi')])

        #models.append([get_model_architecture('lstm'), get_num_units(400), get_num_layers(1), get_dropout_prob(0.9), get_batch(20), get_pretrain(True), get_summaries_dir_ckpt('ptlstm_400_drop_0.9')])
        #models.append([get_model_architecture('attn'), get_num_units(200), get_num_layers(1), get_dropout_prob(0.9), get_batch(100), get_summaries_dir_ckpt('attn_200_drop_0.9')])
        #models.append([get_model_architecture('conv1d'), get_dropout_prob(0.9), get_batch(100), get_summaries_dir_ckpt('conv1d_drop_0.9_clipped')])

        i=1
        for model in models:
            print(bcolors.OKBLUE + bcolors.BOLD + 'model : ' + str(i) + '/' + str(len(models)) + bcolors.ENDC)
            run(model)
            i+=1
