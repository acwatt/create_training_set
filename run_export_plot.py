import os
import time
from datetime import datetime
import subprocess as sp
import shutil
import edit_config as EC

# MODE: 1=drone, 2=satellite, 3=drone+satellite
MODE = 3
# LABELS: 1=custom labels, 2=original labels
#LABLES = 1
FORECAST_ONLY = False

_mode_dict = {1: 'drone_training2',
              2: 'satellite_training3',
              3: 'combined_training',
              4:'drone_angled',
              5:'sat_test',
              6: 'drone_angled_base',
              7: 'satellite_training_base',
              8: 'combined_training_base'}
_translation_dict = {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2}
_label_dict = {1:'bee_label_map.pbtxt',
               2:'mscoco_label_map.pbtxt'}
MODEL_FOLDER_NAME = _mode_dict[MODE]
LABELS = _translation_dict[MODE]
LABEL_FILE_NAME = _label_dict[LABELS]
CREATE_TF_RECORDS = 1
ORIGINAL_CATEGORIES = LABLES - 1 # 1 or 0 to pass to TF record generator to use new or COCO categories
#MINS_RUNNING = 5


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

if __name__ == "__main__":
    if FORECAST_ONLY:
        training_path = 'C:/Users/Administrator/Desktop/create_training_set/' + MODEL_FOLDER_NAME + '/training/'
        checkpoint_numbers = [checkpoint.replace('.meta', '').split('-')[-1] for checkpoint in os.listdir(training_path) \
                        if checkpoint.endswith('.meta')]
        print(checkpoint_numbers)

    # Run multiple back-to-back sessions with different stopping points
    old_steps = '10000'
    for steps in ['100', '200', '500', '1000', '2000', '5000', '10000']: #:checkpoint_numbers
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
            --model_config=ssd_mobilenet_v1_coco.config \
            --convert_images={CREATE_TF_RECORDS} \
            --coco_categories_boolean={ORIGINAL_CATEGORIES}"\
            .format(MODEL_FOLDER=MODEL_FOLDER_NAME, CREATE_TF_RECORDS=CREATE_TF_RECORDS, ORIGINAL_CATEGORIES=ORIGINAL_CATEGORIES)
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
        _run_num = new_max_folder_num(_model_dir_path, str_before_num='model', str_after_num='-')
        #_run_num = '1'
        _export_dir_path = _model_dir_path + 'model' + _run_num + '-' + _chpt_num
        # Command to export checkpoint to frozen graph
        export_model_cmd = "C:/ProgramData/Anaconda3/python " +\
                           "C:/Users/Administrator/Desktop/create_training_set/export_model.py " +\
                           "--checkpoint_num=" + _chpt_num +\
                           " --export_dir_path=" + _export_dir_path +\
                           " --model_name=" + MODEL_FOLDER_NAME
        # Command to use saved frozen graph to forecast on images
        apply_model_cmd = "C:/ProgramData/Anaconda3/python " +\
                          "C:/Users/Administrator/Desktop/create_training_set/apply_model.py " +\
                          "--base_model_path=" + _base_path +\
                          " --model_name=" + MODEL_FOLDER_NAME +\
                          " --exported_dir_path=" + _export_dir_path +\
                          " --min_score_threshold=0.5 " +\
                          " --path_to_labels=" + _path_to_labels +\
                          " --outsample_boolean=True" +\
                          " --resize_width=750" # width of saved forecasted images (to reduce size of output data)
        # EXPORT
        print('\n'+export_model_cmd)
        export_process = sp.Popen(export_model_cmd)
        export_process.wait()
        # FORECAST
        print('\n'+apply_model_cmd)
        apply_model_process = sp.Popen(apply_model_cmd)
        apply_model_process.wait()
        print("\n\nDone forecasting, waitng 5 seconds.")
        time.sleep(5)

        if not FORECAST_ONLY:
            print("tensorboard status: ", monitor_process.poll())
            print("\n\nKilling tensorboard")
            monitor_process.kill()
            monitor_process.wait()
            print("tensorboard status: ", monitor_process.poll())
            while monitor_process.poll() is None:
                print("tensorboard status: ", monitor_process.poll())
                time.sleep(3)
            # Rename training folder
            if 1==0: rename_training_folder(MODEL_FOLDER_NAME, steps)


