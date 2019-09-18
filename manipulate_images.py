import os
import shutil
import numpy as np
import pandas as pd

def split_images(image_directory,num_max,test_split=0.8):
    """
    split a directory of images into a train / test set
    :param image_directory:
    :return:
    """

    train_path = image_directory + 'train/'

    test_path = image_directory + 'test/'

    for image in os.listdir(train_path):
        image_path = train_path + image
        os.remove(image_path)

    for image in os.listdir(test_path):
        image_path = test_path + image
        os.remove(image_path)

    # first get images count
    images = []
    for image in os.listdir(image_directory):
        images.append(image)

    # shuffle list so order doesnt matter and we can sample
    np.random.shuffle(images)

    images = images[:num_max]

    train_index = int(len(images)*test_split)

    train_images = images[:train_index]

    test_images = images[train_index:]

    print("Train length: {}".format(str(len(train_images))))

    print("Test length: {}".format(str(len(test_images))))

    for image in train_images:
        image_path = image_directory + image
        image_train_path = train_path + image
        if os.path.isdir(image_path):
            continue
        shutil.copy(image_path,image_train_path)

    for image in test_images:
        image_path = image_directory + image
        image_test_path = test_path + image
        if os.path.isdir(image_path):
            continue
        shutil.copy(image_path,image_test_path)

def rename_image(directory):
    """
    :param directory:
    :return:
    """

    image_index = 0
    for image in os.listdir(directory):
        if image.split('.')[-1] == 'xml':
            pass
        image_path = directory + image
        new_path = directory + 'train_image_{}.jpg'.format(str(image_index))
        os.rename(image_path,new_path)
        image_index += 1

def add_extension_to_csv(csv_path):
    """
    :param csv_path:
    :return:
    """

    csv_df = pd.read_csv(csv_path)

    csv_df['filename'] = csv_df['filename'].apply(lambda x: x + '.jpg')

    csv_df.to_csv(csv_path)

if __name__ == '__main__':

    run_split_images = 0
    if run_split_images:

        image_directory = 'C:/Users/Grechen/hive_object_detection/images/'

        split_images(image_directory,num_max=50,test_split=0.8)

    run_adjust_csv = 1
    if run_adjust_csv:

        csv_path = 'C:/Users/Grechen/hive_object_detection/data/test_labels.csv'

        add_extension_to_csv(csv_path)