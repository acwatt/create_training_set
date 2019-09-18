import lxml.etree as ET
import os

drc = 'C:/Users/Administrator/Desktop/create_training_set/drone_training_base/images/'
PATH_TO_TRAIN_IMAGES_DIR = drc + 'train/'
PATH_TO_TEST_IMAGES_DIR = drc + 'test/'
PATH_TO_OUTSAMPLE_IMAGES_DIR = drc + 'outsample/'

TRAIN_IMAGE_PATHS = [PATH_TO_TRAIN_IMAGES_DIR + imagefile for imagefile in
                         os.listdir(PATH_TO_TRAIN_IMAGES_DIR) if imagefile.lower().endswith('.xml')]
TEST_IMAGE_PATHS = [PATH_TO_TEST_IMAGES_DIR + imagefile for imagefile in
                        os.listdir(PATH_TO_TEST_IMAGES_DIR) if imagefile.lower().endswith('.xml')]
OUTSAMPLE_IMAGE_PATHS = [PATH_TO_OUTSAMPLE_IMAGES_DIR + imagefile for imagefile in
                             os.listdir(PATH_TO_OUTSAMPLE_IMAGES_DIR) if imagefile.lower().endswith('.xml')]
all_paths = TRAIN_IMAGE_PATHS + TEST_IMAGE_PATHS + OUTSAMPLE_IMAGE_PATHS

for file in all_paths:
    print(file)
    with open(file, 'rb+') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for elem in root.getiterator():
            if elem.text:
                elem.text = elem.text.replace('hive', 'bird')
                elem.text = elem.text.replace('-pallet', '')
                elem.text = elem.text.replace('structure', 'bench')

            if elem.tail:
                elem.tail = elem.tail.replace('hive', 'bird')
                elem.text = elem.text.replace('-pallet', '')
                elem.text = elem.text.replace('structure', 'bench')

        f.seek(0)
        f.write(ET.tostring(tree, encoding='UTF-8', xml_declaration=True))
        f.truncate()






# import re
# import os
# import shutil
#
# drc = 'C:/Users/Administrator/Desktop/create_training_set/drone_training_base/images/'
# pattern = re.compile('hive')
# oldstr = 'hive'
# newstr = 'bird'
#
# for dirpath, dirname, filename in os.walk(drc):#Getting a list of the full paths of files
#     print()
#     print(dirpath, dirname, filename)
#     for fname in filename:
#         path = os.path.join(dirpath, fname) #Joining dirpath and filenames
#         strg = open(path).read() #Opening the files for reading only
#         if re.search(pattern, strg):#If we find the pattern ....
#             print(path, strg)
#             #shutil.copy2(path, backup) #we will create a backup of it
#             strg = strg.replace(oldstr, newstr) #We will create the replacement condition
#             f = open(path, 'w') #We open the files with the WRITE option
#             print("Writing to ", path)
#             f.write(strg) # We are writing the the changes to the files
#             f.close() #Closing the files