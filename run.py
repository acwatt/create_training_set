
import argparse

from xml_to_csv import xml_to_csv_main
from generate_tfrecord import tfrecord_main

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run create training set script')

    parser.add_argument('image_directory', metavar='N', type=str,
                        help='name of training directory')

    args = parser.parse_args()

    # step 1 convert the xml files to csv
    xml_to_csv_main(args.image_directory)

    # step 2 create training tf records
    tfrecord_main(args.image_directory)