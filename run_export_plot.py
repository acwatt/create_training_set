import os
import time
import subprocess as sp

# MODE: 1=drone, 2=satellite, 3=drone+satellite
MODE = 2
# LABELS: 1=custom labels, 2=original labels
LABLES = 1

_mode_dict = {1: 'drone_training2', 2: 'satellite_training_palletsplus', 3: 'combined_training'}
_label_dict = {1:'bee_label_map.pbtxt', 2:'mscoco_label_map.pbtxt'}
MODEL_FOLDER_NAME = _mode_dict[MODE]
LABEL_FILE_NAME = _label_dict[LABLES]
CREATE_TF_RECORDS = 1
#MINS_RUNNING = 5


def _kill(process):
    """Was trying to use a timer to kill the processes after some preset time. But this was unable to kill all
    the child processes created by tensorflow."""
    while process.poll() is None:  # still running
        print('TRYING TO KILL')
        process.kill()


def new_max_folder_num(base_folder, str_before_num='', str_after_num=''):
    """Find highest numbered folder in models/ folder, and return that number + 1 to use in next folder name."""
    return str(max([int(folder.replace(str_before_num, '').split(str_after_num)[0]) for folder in os.listdir(base_folder) \
                    if len(folder.split(str_after_num)) > 1], default=0) + 1)


if __name__ == "__main__":
    # Must run from anaconda prompt

    #conda_cmd = "C:\ProgramData\Anaconda3\Scripts\conda.exe activate tensorflow"

    # Command to start the training process
    run_model_cmd = "C:/ProgramData/Anaconda3/python \
        C:/Users/Administrator/Documents/models/research/object_detection/gms_create_training_set/run.py  \
        --image_directory=C:/Users/Administrator/Desktop/create_training_set/{MODEL_FOLDER} \
        --model_config=ssd_mobilenet_v1_coco.config \
        --convert_images={CREATE_TF_RECORDS}"\
        .format(MODEL_FOLDER=MODEL_FOLDER_NAME, CREATE_TF_RECORDS=CREATE_TF_RECORDS)
    # Command to start tensorboard to monitor training
    monitor_cmd = "tensorboard --logdir=C:/Users/Administrator/Desktop/create_training_set/{MODEL_FOLDER}/training"\
                .format(MODEL_FOLDER=MODEL_FOLDER_NAME)
    # Change to object_detection directory to run the commands
    os.chdir('C:/Users/Administrator/Documents/models/research/object_detection/')
    print(run_model_cmd)
    ## run training
    print('\n' + os.getcwd())
    #conda_env_process = sp.Popen(conda_cmd)
    print('\n'+run_model_cmd)
#    run_model_process = sp.Popen(run_model_cmd)
    ## run monitoring
#    time.sleep(10)
    print('\n'+monitor_cmd)
#    monitor_process = sp.Popen(monitor_cmd)
    ## when finished training, export model
    #print('time.sleep(%i)'%(60*MINS_RUNNING))
    #time.sleep(60*MINS_RUNNING)
    #_kill(run_model_process)
#    run_model_process.wait()

    # path to training directory of model in use
    _training_dir = "C:/Users/Administrator/Desktop/create_training_set/{MODEL_FOLDER}/training/"\
                    .format(MODEL_FOLDER=MODEL_FOLDER_NAME)
    # path to labels file being used for class labels
    _path_to_labels = _training_dir + LABEL_FILE_NAME
    ## checkpoint # equal to max of checkpoints in training
    _chpt_num = str(max([int(chpt.replace('model.ckpt-', '').split('.index')[0]) \
                         for chpt in os.listdir(_training_dir) if len(chpt.split('.index')) > 1], default=0))
    ## export that checkpoint to models folder
    _base_path = 'C:/Users/Administrator/Desktop/create_training_set/' + MODEL_FOLDER_NAME + '/'
    _model_dir_path = _base_path + '/models/'
    _run_num = new_max_folder_num(_model_dir_path, str_before_num='model', str_after_num='-')
    _run_num = '1'
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
                      " --min_score_threshold=0.04 " +\
                      " --path_to_labels=" + _path_to_labels +\
                      " --outsample_boolean=True" # forecast on outsample folder
    # EXPORT
    print('\n'+export_model_cmd)
#    export_process = sp.Popen(export_model_cmd)
#    export_process.wait()
    # FORECAST
    print('\n'+apply_model_cmd)
    apply_model_process = sp.Popen(apply_model_cmd)

    # Next, pass model folder name apply_model
    # have apply_model add test_ and train_ to beg of pic names,
    #   and place in folder with date-name_of_model-iterations in output predictions



