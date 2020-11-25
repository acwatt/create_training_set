#!C:\ProgramData\Anaconda3 python3
# Author: Aaron Kratzer, 2019
# Purpose:  to replace items in the model config. files, with extra functions
#           to replace labels in XML files if the user wants to rename labels.
import fileinput
import os
import time

def list_files(extension):
    """list all files in current directory with extension"""
    return [file for file in os.listdir() if len(file.lower().split(extension.lower()))>1]


def replace_text_in_file(filename, text_to_search, replacement_text):
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end='')


def xml_find_filename(xmlfile):
    myfile = open(xmlfile, "rt")  # open lorem.txt for reading text
    contents = myfile.read()  # read the entire file into a string
    filename = contents.split('filename')[1].replace('>','').replace('</','')
    myfile.close()  # close the file
    return filename


def replace_xml_filename(xml_filename):
    old_filename = xml_find_filename(xml_filename) # filename stored in xml (of a jpg or png)
    old_filename_ext = old_filename.split('.')[-1] # extension of filename ('jpg' or 'png')
    new_filename = ".".join(xml_filename.split('.')[:-1]) + '.' + old_filename_ext
    print("changing filename param from ", old_filename, " to ", new_filename)
    with fileinput.FileInput(xml_filename, inplace=True) as file:
        #time.sleep(5)
        for line in file:
            print(line.replace(old_filename, new_filename), end='')


def xml_filename_directory_replace(directory_path):
    """replaces filenames in xmls in directory_path (before extension) with filename of xml (before extension).
    If name of xml-img pair has been changed, this will edit the xml filename to match change

    Example: xml name: sat_1234.xml, old xml 'filename': DIJ_1234.JPG, replaced with: sat_1234.JPG
    Keeps extension, replaces front part of filename"""
    os.chdir(directory_path)
    files = list_files('.xml')
    print("Iterating through files: ", files)
    for file in files:
        print("working on file ", file)
        replace_xml_filename(file)
        print("Replaced filename in ", file)


def xml_categories_replace(filename, category_listdicts):
    """replaces categories in file with values in category_dict"""
    for row in category_listdicts:
        [key, value] = row
        print("replacing ", key, " with ", value)
        with fileinput.FileInput(filename, inplace=True) as file:
            for line in file:
                print(line.replace(key, value), end='')


def xml_directory_categories_replace(directory_path, category_listdicts):
    os.chdir(directory_path)
    files = list_files('.xml')
    print("Iterating through files: ", files)
    for file in files:
        print("working on file ", file)
        xml_categories_replace(file, category_listdicts)
        print("Replaced categories in ", file)


if __name__ == "__main__":
    category_listdicts = [['hive-pallet','bird'], ['hive','bird'], ['structure','bench']] # ordered dictionary, since hive-pallet must be replace before hive
    model_name = 'combined_training_base'
    for image_type in ['train', 'test', 'outsample']:
        directory = 'C:/Users/Administrator/Desktop/create_training_set/' + model_name + '/images/' + image_type
        print(directory)
        xml_directory_categories_replace(directory, category_listdicts)