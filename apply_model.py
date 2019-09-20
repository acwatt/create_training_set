###################################################
## Apply the frozen inference graph (trained model) to
##       pictures in train and test folders
###################################################

#                 SETTINGS
###########################################################
# What model to use
#MODEL_FOLDER = 'satellite_training2'
#CHECKPOINT_NAME = 'model7-1396'
#OUTPUT_FOLDER = 'C:/Users/Administrator/Desktop/create_training_set/output_predictions/sat_4categories_2019-09-16' #check lower sections

#FONTSIZE = 32
#RUN_OG_SSD = 3



###########################################################

#                 IMPORTSS
###########################################################
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import argparse
from datetime import datetime


# This is needed since the notebook was stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
###########################################################


#MODEL_FILE = INFERENCE_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
"""
BASE_PATH = 'C:/Users/Administrator/Desktop/create_training_set/' + MODEL_FOLDER + '/'
if not RUN_OG_SSD:
    CHECKPOINT_NAME = input('Enter the folder of the exported model: ')
    INFERENCE_NAME = BASE_PATH + 'models/' + CHECKPOINT_NAME
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = INFERENCE_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(BASE_PATH + 'training', 'bee_label_map.pbtxt')


# If running OG SSD Mobilenet
if RUN_OG_SSD == 1:
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    PATH_TO_FROZEN_GRAPH = 'C:/Users/Administrator/Documents/models/research/object_detection/' + MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('C:/Users/Administrator/Documents/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

if RUN_OG_SSD == 2: #Drone model on satellite pics
    MODEL_NAME = 'drone_training2'
    PATH_TO_FROZEN_GRAPH = 'C:/Users/Administrator/Desktop/create_training_set/drone_training2/models/model8-1000/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('C:/Users/Administrator/Desktop/create_training_set/drone_training2/training', 'bee_label_map.pbtxt')
    BASE_PATH = 'C:/Users/Administrator/Desktop/create_training_set/satellite_training2/' #for images
    OUTPUT_FOLDER = 'C:/Users/Administrator/Desktop/create_training_set/output_predictions/drone_on_sat'

if RUN_OG_SSD == 3: #Satellite model on sat pics
    MODEL_NAME = 'satellite_training2'
    PATH_TO_FROZEN_GRAPH = 'C:/Users/Administrator/Desktop/create_training_set/satellite_training2/models/model4-10000/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('C:/Users/Administrator/Desktop/create_training_set/satellite_training2/training', 'bee_label_map.pbtxt')
    BASE_PATH = 'C:/Users/Administrator/Desktop/create_training_set/satellite_training2/' #for images
    OUTPUT_FOLDER = 'C:/Users/Administrator/Desktop/create_training_set/output_predictions/sat_4categories_2019-09-16'

if RUN_OG_SSD == 4: #base model + training with original categories on drone pics
    MODEL_NAME = 'drone_training_base'
    PATH_TO_FROZEN_GRAPH = 'C:/Users/Administrator/Desktop/create_training_set/drone_training_base/models/model1-2000/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('C:/Users/Administrator/Desktop/create_training_set/drone_training_base/training', 'mscoco_label_map.pbtxt')
    BASE_PATH = 'C:/Users/Administrator/Desktop/create_training_set/drone_training_base/' #for images
    OUTPUT_FOLDER = 'C:/Users/Administrator/Desktop/create_training_set/output_predictions/drone_OGcategories'

"""



# This is needed to display the images.
#%matplotlib inline

def resize_images_in_dir(directory_path, resize_width, replace=True):
    """Resizes all images in directory_path to a width of resize_width, keeping aspect ratio

    Usage:
        resize_images_in_dir("C:/Users/Administrator/imagesfolder", 500)

    Args:
        directory_path: full path to directory
        resize_width: width in pixels. 500 seems sufficient for large images
        replace: when True, replaces all images in directory. When False, just makes resized copies"""




def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])

                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def list_of_images_to_forecast(model_folder_path, ALL_IMAGES=True, outsample_yn=False):
    BASE_PATH = model_folder_path
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TRAIN_IMAGES_DIR = BASE_PATH + '/images/train/'
    PATH_TO_TEST_IMAGES_DIR = BASE_PATH + '/images/test/'
    PATH_TO_OUTSAMPLE_IMAGES_DIR = BASE_PATH + '/images/outsample/'

    if not ALL_IMAGES:
        train_images = ['DJI_0017.JPG']  # , 'DJI_0516.JPG']
        test_images = ['DJI_0176.JPG']  # , 'DJI_0413.JPG']
        outsample_images = ['DJI_0017.JPG', 'DJI_0516_adj.JPG']

        TRAIN_IMAGE_PATHS = [(image + '.png', os.path.join(PATH_TO_TRAIN_IMAGES_DIR, image), 'TRAIN') for image in
                             train_images]
        TEST_IMAGE_PATHS = [(image + '.png', os.path.join(PATH_TO_TEST_IMAGES_DIR, image), 'TEST') for image in
                            test_images]
        OUTSAMPLE_IMAGE_PATHS = [(image + '.png', os.path.join(PATH_TO_OUTSAMPLE_IMAGES_DIR, image), 'OUTSAMPLE') for
                                 image in outsample_images]

    # create overlay predictions for all images in folders
    if ALL_IMAGES:
        TRAIN_IMAGE_PATHS = [(imagefile.split('\\')[-1].split('/')[-1], PATH_TO_TRAIN_IMAGES_DIR + imagefile, 'TRAIN')
                             for imagefile in os.listdir(PATH_TO_TRAIN_IMAGES_DIR) if imagefile.lower().endswith('.jpg')]
        TEST_IMAGE_PATHS = [(imagefile.split('\\')[-1].split('/')[-1], PATH_TO_TEST_IMAGES_DIR + imagefile, 'TEST') for
                            imagefile in os.listdir(PATH_TO_TEST_IMAGES_DIR) if imagefile.lower().endswith('.jpg')]
        OUTSAMPLE_IMAGE_PATHS = [
            (imagefile.split('\\')[-1].split('/')[-1], PATH_TO_OUTSAMPLE_IMAGES_DIR + imagefile, 'OUTSAMPLE') for
            imagefile in
            os.listdir(PATH_TO_OUTSAMPLE_IMAGES_DIR) if imagefile.lower().endswith('.jpg')]

    TOTS_IMAGES = TRAIN_IMAGE_PATHS + TEST_IMAGE_PATHS
    if outsample_yn: TOTS_IMAGES += OUTSAMPLE_IMAGE_PATHS
    print("\n\nIMAGES TO FORECAST ON:")
    for imagefile in TOTS_IMAGES: print(imagefile)
    return(TOTS_IMAGES)


def forecast_on_imagepaths(image_paths_list,
                           export_dir_path,
                           path_to_labels,
                           output_dir_path,
                           model_name,
                           steps,
                           min_score_threshold=0.5,
                           fontsize_increment=8,
                           fig_scale=1.0,
                           resize_width=500):
    """
    Forecasts boxes & classes onto images in image_paths_list using model in export_dir_path.
    :param image_paths_list: list of tuples of (image_name, image_path, image_type) where image_type={TEST, TRAIN}
    :param export_dir_path: path to exported frozen graph from model
    :param path_to_labels: path to .pbtxt file with class labels for predicted boxes
    :param output_dir_path: path to directory where forecasted images will be saved
    :param min_score_threshold: box predictions larger than this will show on forecasted image
    :param fontsize_increment: >8 will make font larger on images > 1200 px wide
    :param fig_scale: >1 will make final images larger
    :param resize_width: pixel width to resize images to when saving. Set to 0 to save in original resolution
    :return: None (saves images to output_directory
    """
    path_to_frozen_graph = export_dir_path + '/frozen_inference_graph.pb'
    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    all_images = image_paths_list
    # make output directory
    try:
        os.mkdir(output_dir_path)
    except FileExistsError:
        print("*******************  Output Directory already exists!!!  *******************")
    # Size, in inches, of the output images.
    image_size = (int(20*fig_scale), int(12*fig_scale))
    i = 1
    length = len(all_images)
    for (image_name, image_path, image_type) in all_images:
        print('*********************** starting %s of %s ******************************'%(i,length))
        i+=1
        print(image_path)
        print(image_type)
        image = Image.open(image_path).convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        print(image.getdata())
        #np.array(image.getdata())
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # fontize dependent on the width of the image in pixels
        fontsize = int(((image.width - 1200) / 700) * fontsize_increment + 24) #24 is the base font size, increasing in increments of 8pts
        linethickness = int(((image.width - 1200) / 700) + 4)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=linethickness, min_score_thresh=min_score_threshold, FONTSIZE=fontsize)
        plt.figure(figsize=image_size)
        #plt.imshow(image_np)
        image_split = image_name.split('.') # pop off extension
        image_name, image_ext = ".".join(image_split[:-1]), '.' + image_split[-1] # save filename and ext separately
        image_save_path = output_dir_path +'/'+ image_type +'-'+ image_name +'-'+ model_name +'-'+ steps + image_ext
        # Save image
        plt.imsave(image_save_path, image_np)
        plt.close('all') # must close, otherwise all figures stay open until done with list of images
        print('IMAGE SAVED TO: ', image_save_path)
        # Resize image
        if resize_width > 0:
            img = Image.open(image_save_path)
            wpercent = (resize_width / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((resize_width, hsize), Image.ANTIALIAS)
            img.save(image_save_path)
            print('IMAGE RESIZED TO: ', resize_width)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Use a frozen graph of a model to forecast on folders of images.')
    parser.add_argument('--exported_dir_path', metavar='path/to/MODEL#_####', type=str,
                        help='path of model folder saved from export model (in models folder)')
    parser.add_argument('--base_model_path', metavar='path/to/main_model_dir', type=str,
                        help='path to directory of model being trained (C:/Users/Administrator/Desktop/create_training_set/drone_training2)')
    parser.add_argument('--model_name', metavar='DIRECTORY NAME', type=str,
                        help='name of model directory, like drone_training2')
    parser.add_argument('--min_score_threshold', metavar='NUM (0-1)', type=float,
                        help='Give a number between 0 and 1 for the probability threshold to show predicted boxes',
                        default=0.5)

    parser.add_argument('--output_dir_path', metavar='DIRECTORY PATH', type=str,
                        help='Type full path of directory to save forecasted images to (C:/Users/Administrator/Desktop/create_training_set/output_predictions/sat_4categories_2019-09-16)',
                        default="")
    parser.add_argument('--path_to_labels', metavar='PATH', type=str,
                        help='Full path to .pbtxt labels file')
    parser.add_argument('--outsample_boolean', metavar='BOOLEAN', type=bool,
                        help='True=forecast on outsample folder, False=do not forecast on outsample folder.',
                        default=False)
    parser.add_argument('--resize_width', metavar='Pixels', type=int,
                        help='width in pixels of saved forecasted images.',
                        default=0)
    args = parser.parse_args()


    _base_path = args.base_model_path
    _list_of_image_paths = list_of_images_to_forecast(_base_path, outsample_yn=args.outsample_boolean)



    if 1 == 1:
        train_list = ['30.014_-94.924.JPG',
                      '39.135_-77.348.JPG',
                      '44.333_-101.066.JPG',
                      'DJI_0051.JPG',
                      'DJI_0077.JPG',
                      'DJI_0173.JPG',
                      'DJI_0528.JPG']
        temp_list = []
        for t in _list_of_image_paths:
            if t[-1] == "OUTSAMPLE" or t[-1] == "TEST":
                temp_list.append(t)
                print("image added: ",t[-1], t[0])
            elif t[-1] == "TRAIN" and t[0] in train_list:
                temp_list.append(t)
                print("image added: ", t[-1], t[0])
            else:
                print("image not added: ", t[-1], t[0])
        _list_of_image_paths = temp_list

    if 1 == 0:
        _list_of_image_paths = [('30.014_-94.924.JPG', 'C:/Users/Administrator/Desktop/create_training_set/satellite_training3//images/train/30.014_-94.924.JPG', 'TRAIN'),
                                ('30.047_-94.712.JPG', 'C:/Users/Administrator/Desktop/create_training_set/satellite_training3//images/train/30.047_-94.712.JPG', 'TRAIN')]



    # Name the output directory
    _today = datetime.now().strftime("%Y%m%d_%H%M")
    _steps = args.exported_dir_path.split('-')[-1]
    prob_threshold = args.min_score_threshold * 100
    if prob_threshold <10: prob_threshold = '0'+str(int(args.min_score_threshold*100))
    else: prob_threshold = str(int(prob_threshold))
    _save_dir_name = _today + '-' + args.model_name + '-' + _steps + '-' + prob_threshold
    if args.output_dir_path == "":
        _output_dir_path = "C:/Users/Administrator/Desktop/create_training_set/output_predictions/" + _save_dir_name
    else:
        _output_dir_path = args.output_dir_path

    forecast_on_imagepaths(_list_of_image_paths,    # list of tuples: (image_name, image_path, image_type)
                           args.exported_dir_path,        # path to exported frozen graph directory
                           args.path_to_labels,         # path to .pbtxt file with labels in it
                           _output_dir_path,        # path to new output directory where images will be saved
                           args.model_name,
                           _steps,
                           resize_width=args.resize_width,
                           min_score_threshold=args.min_score_threshold,
                           fontsize_increment=8,
                           fig_scale=.2)

