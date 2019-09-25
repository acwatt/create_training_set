import os
import time
from datetime import datetime
import time
import subprocess as sp
import shutil
import edit_config as EC
import apply_model

# MODE: see _mode_dict below
MODE = 5 #int
# LABELS: 1=custom labels, 2=original coco labels, 3=first 16 original coco lables
FORECAST_ONLY = True #bool, exports models at checkpoints in _step_list, then forecasts on all picture
USE_CHECKPOINT_NUMBERS_AS_STEPS = False #bool, only is used when FORECAST_ONLY is True
EXPORT_MODEL = False #bool
PROB_THRESHOLD = 0.01 #float
MODEL_NUM = 1#None #for model3-10000, use 3, int
TEST_FORECAST = False #bool, only use 2 training images, and no test
USE_RCNN = 0 #int, 0 or 1

_mode_dict = {1: {'name':'drone_training2',         'description': 'drone pics, new 4 categories'},
              2: {'name':'satellite_training3',     'description': 'same sat pics as sat_2 + structures and cars labeled, new 4 categories'},
              3: {'name':'combined_training',       'description': 'both satellite and aerial+angled drone pics, new 4 categories'},
              4: {'name':'drone_angled',            'description': 'All aerial drone pics + more angled drone pics, new 4 categories'},
              5: {'name':'sat_test',                'description': 'used to test code changes, using faster_r-cnn'},
              6: {'name':'drone_angled_base',       'description': 'drone pics aerial+angled, only 3 of the OG COCO categories'},
              7: {'name':'satellite_training_base', 'description': 'all 90 OG COCO categories'},
              8: {'name':'combined_training_base',  'description': 'satellite and drone angled+aerial pics, all 90 OG COCO categories'},
              9: {'name':'satellite_training_base2','description': 'satellite pics, only 3 of the OG COCO categories'}}

steps_to_replace  = '10000'
_step_list = ['0'] #'3', '100', '200', '500', '1000', '2000', '5000', '10000'
#_step_list = ['10000']
_translation_dict = {1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:3, 9:3}
_label_dict = {1:'bee_label_map.pbtxt', # uses
               2:'mscoco_label_map.pbtxt',
               3:'mscoco_label_map_3cats.pbtxt'}
_config_dict = {0:'ssd_mobilenet_v1_coco.config',
                1:'faster_rcnn_nas_coco.config'}
MODEL_FOLDER_NAME = _mode_dict[MODE]['name'] #stb
MODEL_DESCRIPTION = _mode_dict[MODE]['name'] +': '+ _mode_dict[MODE]['description']
LABELS = _translation_dict[MODE] #2
LABEL_FILE_NAME = _label_dict[LABELS] #mscoco
CREATE_TF_RECORDS = 1
_label_to_cat_dict = {1:False, 2:True, 3:True}
ORIGINAL_CATEGORIES = _label_to_cat_dict[LABELS] #pass to generate_tfrecord.py to use new or COCO categories (1)
MODEL_CONFIG = _config_dict[USE_RCNN]

def _kill(process):
    """Was trying to use a timer to kill the processes after some preset time. But this was unable to kill all
    the child processes created by tensorflow."""
    while process.poll() is None:  # still running
        print('TRYING TO KILL')
        process.kill()


def new_max_folder_num(base_folder, str_before_num='', str_after_num=''):
    """Find highest numbered folder in models/ folder, and return that number + 1 to use in next folder name."""
    nums = [int(folder.replace(str_before_num, '').split(str_after_num)[0]) for folder in os.listdir(base_folder) \
                    if len(folder.split(str_after_num)) > 1]
    print('\n\nExisting Models: ', nums)
    return str(max(nums, default=0) + 1)


def new_training_folder(model_folder_name, base_path='C:/Users/Administrator/Desktop/create_training_set/'):
    """Copies training-default/ to training/ for use for a new round of training"""
    training_folder_path = base_path + model_folder_name + '/training'
    shutil.copytree(training_folder_path+'-default', training_folder_path)


def rename_training_folder(model_folder_name, steps,
                           base_path='C:/Users/Administrator/Desktop/create_training_set/'):
    """Renames the training folder in this model to datetime_training_steps"""
    training_folder_path = base_path + model_folder_name + '/training'
    today = datetime.now().strftime("%Y%m%d_%H%M")
    new_path = base_path + model_folder_name + '/' + today + '-training-' + steps
    os.rename(training_folder_path, new_path)


def create_summary_txt(time_begin, time_end, model_details, image_numbers, output_dir):
    """create a _specifications.txt file in last forecast directory"""
    string = []
    # Model name
    string.append('Model: ' + model_details + '\n\n')
    # Datetime ran
    string.append('Date of training: ' + str(datetime.now().strftime("%Y-%m-%d %H:%M")) + '\n\n')
    # Time to complete
    time_total_hr = (time_end - time_begin) / 3600
    string.append('Time for completion (including forecasting): ' + str(round(time_total_hr, 4)) + ' hours\n\n')
    # Images
    string.append('Images:\n' \
                  '    Train: ' + str(image_numbers['train']) + '\n' \
                  '    Test:  ' + str(image_numbers['test']) + '\n' \
                  '    Outsample: ' + str(image_numbers['outsample']))
    # Write _specifications.txt file
    out_filename = output_dir + '/_specifications.txt'
    with open(out_filename, 'a') as out_file:
        out_file.writelines(string)


if __name__ == "__main__":
    time_begin = time.time()
    if FORECAST_ONLY:
        training_path = 'C:/Users/Administrator/Desktop/create_training_set/' + MODEL_FOLDER_NAME + '/training/'
        checkpoint_numbers = [checkpoint.replace('.meta', '').split('-')[-1] for checkpoint in os.listdir(training_path) \
                        if checkpoint.endswith('.meta')]
        print(checkpoint_numbers)
        if USE_CHECKPOINT_NUMBERS_AS_STEPS: _step_list = checkpoint_numbers

    # Run multiple back-to-back sessions with different stopping points
    old_steps = steps_to_replace
    for steps in _step_list:
        print("\n\nBeginning model run with "+steps+" steps for model "+MODEL_FOLDER_NAME)
        # Must run from anaconda prompt

        configfile = 'C:/Users/Administrator/Desktop/create_training_set/' + \
                     MODEL_FOLDER_NAME + '/training/ssd_mobilenet_v1_coco.config'
        if not FORECAST_ONLY:
            # create new training directory
            if 1==0: new_training_folder(MODEL_FOLDER_NAME)
            # edit number of steps in config
            EC.replace_text_in_file(configfile, 'num_steps: ' + old_steps, 'num_steps: ' + steps)
            old_steps = steps

        # Command to start the training process
        run_model_cmd = "C:/ProgramData/Anaconda3/python \
            C:/Users/Administrator/Documents/models/research/object_detection/gms_create_training_set/run.py  \
            --image_directory=C:/Users/Administrator/Desktop/create_training_set/{MODEL_FOLDER} \
            --model_config={MODEL_CONFIG} \
            --convert_images={CREATE_TF_RECORDS} \
            --coco_categories_boolean={ORIGINAL_CATEGORIES}"\
            .format(MODEL_FOLDER=MODEL_FOLDER_NAME, MODEL_CONFIG=MODEL_CONFIG, CREATE_TF_RECORDS=CREATE_TF_RECORDS, ORIGINAL_CATEGORIES=ORIGINAL_CATEGORIES)
        CREATE_TF_RECORDS = 0 # after first session, use existing TF records
        # Command to start tensorboard to monitor training
        monitor_cmd = "tensorboard --logdir=C:/Users/Administrator/Desktop/create_training_set/{MODEL_FOLDER}/training"\
                    .format(MODEL_FOLDER=MODEL_FOLDER_NAME)
        # Change to object_detection directory to run the commands
        os.chdir('C:/Users/Administrator/Documents/models/research/object_detection/')
        print(run_model_cmd)
        ## run training
        print('\n' + os.getcwd())
        print('\n'+run_model_cmd)
        print('\n'+monitor_cmd)
        if not FORECAST_ONLY:
            run_model_process = sp.Popen(run_model_cmd)
            time.sleep(120)
            monitor_process = sp.Popen(monitor_cmd)
            run_model_process.wait()
            print("tensorboard status: ", monitor_process.poll())
            print("\n\nKilling tensorboard")
            monitor_process.kill()
            monitor_process.wait()
            print("tensorboard status: ", monitor_process.poll())
            while monitor_process.poll() is None:
                print("tensorboard status: ", monitor_process.poll())
                time.sleep(3)

        # path to training directory of model in use
        _training_dir = "C:/Users/Administrator/Desktop/create_training_set/{MODEL_FOLDER}/training/"\
                        .format(MODEL_FOLDER=MODEL_FOLDER_NAME)
        # path to labels file being used for class labels
        _path_to_labels = _training_dir + LABEL_FILE_NAME
        ## checkpoint # equal to max of checkpoints in training
        if not FORECAST_ONLY: _chpt_num = str(max([int(chpt.replace('model.ckpt-', '').split('.index')[0]) \
                             for chpt in os.listdir(_training_dir) if len(chpt.split('.index')) > 1], default=0))
        if FORECAST_ONLY: _chpt_num = steps
        ## export that checkpoint to models folder
        _base_path = 'C:/Users/Administrator/Desktop/create_training_set/' + MODEL_FOLDER_NAME + '/'
        _model_dir_path = _base_path + '/models/'
        if MODEL_NUM == None: _run_num = new_max_folder_num(_model_dir_path, str_before_num='model', str_after_num='-')
        else: _run_num = str(MODEL_NUM)
        _export_dir_path = _model_dir_path + 'model' + _run_num + '-' + _chpt_num
        # Command to export checkpoint to frozen graph
        export_model_cmd = "C:/ProgramData/Anaconda3/python " +\
                           "C:/Users/Administrator/Desktop/create_training_set/export_model.py " +\
                           "--checkpoint_num=" + _chpt_num +\
                           " --export_dir_path=" + _export_dir_path +\
                           " --model_name=" + MODEL_FOLDER_NAME
        # EXPORT
        print('\n'+export_model_cmd)
        if EXPORT_MODEL: export_process = sp.Popen(export_model_cmd)
        if EXPORT_MODEL: export_process.wait()
        # FORECAST
        if TEST_FORECAST:
            image_path_list = ([('30.014_-94.924.JPG', 'C:/Users/Administrator/Desktop/create_training_set/satellite_training3//images/train/30.014_-94.924.JPG', 'TRAIN'),
                                ('30.047_-94.712.JPG', 'C:/Users/Administrator/Desktop/create_training_set/satellite_training3//images/train/30.047_-94.712.JPG', 'TRAIN')],
                               {'train': 2, 'test': 0,'outsample': 0})
        else: image_path_list = None

        image_numbers, output_dir_path = apply_model.apply_model(_export_dir_path, #exported_dir_path
                                _base_path, #base_model_path
                                MODEL_FOLDER_NAME, #model_name
                                _path_to_labels, #path_to_labels
                                min_score_threshold=PROB_THRESHOLD, #0.5
                                outsample_boolean=True, #False
                                resize_width=750, #0
                                image_path_list=image_path_list) # to use a short list of 2 images for testing forecast
        print("\n\nDone forecasting, waitng 5 seconds.")
        time.sleep(5)



        # how long did it take
        time_end = time.time()

        # write a summary txt to put in last forecast folder
        create_summary_txt(time_begin, time_end, MODEL_DESCRIPTION, image_numbers, output_dir_path)
