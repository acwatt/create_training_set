## Exporting a trained model to the models folder of current training set

import os
import argparse

## Edit These
#MODEL_NAME='drone_training2-testobjects'
#CHPT_NUM='422'
BASE_PATH = 'C:/Users/Administrator/Desktop/create_training_set/'
INPUT_TYPE = 'image_tensor'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run create training set script')
    parser.add_argument('--checkpoint_num', metavar='NUM', type=str,
                        help='number of checkpoint to use')
    parser.add_argument('--export_dir_path', metavar='DIRECTORY NAME', type=str,
                        help='Name of directory to export model graph to, model#_#-of-steps (like model2_1200)')
    parser.add_argument('--model_name', metavar='DIRECTORY', type=str,
                        help='name of directory of model being trained')
    args = parser.parse_args()

    CHPT_NUM = args.checkpoint_num
    MODEL_NAME = args.model_name
    _export_dir_path = args.export_dir_path
    # create directory to export model frozen graph to
    os.system('mkdir ' + _export_dir_path)

    # From tensorflow/models/research/
    BASE_PATH = BASE_PATH + MODEL_NAME
    PIPELINE_CONFIG_PATH = BASE_PATH + '/training/pipeline.config'
    TRAINED_CKPT_PREFIX = BASE_PATH + '/training/model.ckpt-' + CHPT_NUM

    # create command to take checkpoint and export to a frozen graph in EXPORT_DIR
    run_command = "C:/ProgramData/Anaconda3/python \
        C:/Users/Administrator/Documents/models/research/object_detection/export_inference_graph.py \
        --input_type={INPUT_TYPE} \
        --pipeline_config_path={PIPELINE_CONFIG_PATH} \
        --trained_checkpoint_prefix={TRAINED_CKPT_PREFIX} \
        --output_directory={export_dir_path}"\
        .format(INPUT_TYPE=INPUT_TYPE, PIPELINE_CONFIG_PATH=PIPELINE_CONFIG_PATH,
                                     TRAINED_CKPT_PREFIX=TRAINED_CKPT_PREFIX, export_dir_path=_export_dir_path)
    print(run_command)
    # run command
    os.system(run_command)

'''
satellite 8/28=4637, 

drone 8/27=?, 8/30=1396, 9/3-classestest=600, 


'''