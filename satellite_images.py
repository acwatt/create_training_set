import os
import pandas as pd
import requests
import time

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

GOOGLE_API_URL = '''https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size_x}x{size_y}&scale={scale}&format={format}&maptype=satellite&markers=color:blue%7Clabel:S%7C40.702147,-74.015794&markers=color:green%7Clabel:G%7C40.711614,-74.012318&markers=color:red%7Clabel:C%7C40.718217,-73.998284&key={key}'''
API_KEY = 'AIzaSyDMC96q2fDfin6bjZ096bCZwP3ZQR2g_gI'
ZOOM_LEVEL = 20
SCALE = 4
SIZE_X = 1000
SIZE_Y = 1000
FORMAT = 'JPG'


def _get_if_exist(data, key):
    if key in data:
        return data[key]

    return None


def _convert_to_degress(value):
    """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
    d0 = value[0][0]
    d1 = value[0][1]
    d = float(d0) / float(d1)

    m0 = value[1][0]
    m1 = value[1][1]
    m = float(m0) / float(m1)

    s0 = value[2][0]
    s1 = value[2][1]
    s = float(s0) / float(s1)

    return d + (m / 60.0) + (s / 3600.0)


def create_images_from_excel(excel_path ,output_dir):
    """
    Function that takes in a path to an excel doc that
    contains lat, lon for locations of bee hives.

    this assumes excel structure created in lat_lon_from_images

    out_dir: path where images will get dumped to
    """

    excel_df = pd.read_excel(excel_path)

    for index, row in excel_df.iterrows():

        time.sleep(0.25)  # dont hit google too hard

        lat = row['Latitude']
        lon = row['Longitude']
        name = row['Image']

        a_url = GOOGLE_API_URL.format(latitude=lat, longitude=lon, format=FORMAT,zoom=ZOOM_LEVEL, key=API_KEY,scale=SCALE,size_x=SIZE_X,size_y=SIZE_Y)
        content = requests.get(a_url).content

        image_path = output_dir + name + '.' + FORMAT
        with open(image_path, 'wb') as f:
            f.write(content)

        f.close()


def get_exif_data(image):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value

    return exif_data


def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""
    lat = None
    lon = None

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]

        gps_latitude = _get_if_exist(gps_info, "GPSLatitude")
        gps_latitude_ref = _get_if_exist(gps_info, 'GPSLatitudeRef')
        gps_longitude = _get_if_exist(gps_info, 'GPSLongitude')
        gps_longitude_ref = _get_if_exist(gps_info, 'GPSLongitudeRef')

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degress(gps_latitude)
            if gps_latitude_ref != "N":
                lat = 0 - lat

            lon = _convert_to_degress(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon

    return lat, lon


def lat_lon_from_images(image_directory,output_path):

    """
    Create excel file image_name,lat,lon
    all images in image_directory will be used

    :return: None
    """

    lat_list = []

    for file in os.listdir(image_directory):

        file_path = image_directory + file

        image = None

        try:
            image = Image.open(file_path)
            image.verify()

        except Exception:
            continue # must not be an image

        exif_data = get_exif_data(image)

        lat, lon = get_lat_lon(exif_data)

        if type(lat) == float and type(lon) == float:

            lat_list.append([file.split('.')[0],lat,lon])

        else:
            print(file + " had no valid lat lon")

    lat_df = pd.DataFrame(lat_list,columns=["Image","Latitude","Longitude"])

    lat_df.to_excel(output_path)




if __name__ == '__main__':

    image_directory = 'C:/Users/Administrator/Desktop/create_training_set/texas_training2/images/train/'
    output_path = 'C:/Users/Administrator/Desktop/create_training_set/satellite_training2/images/lat_lon_adj2.xlsx'
    manual_sat_latlons_path = 'C:/Users/Administrator/Desktop/create_training_set/satellite_training2/images/new-sat-download-latlons.xlsx'

    #lat_lon_from_images(image_directory,output_path)

    output_image_path = 'C:/Users/Administrator/Desktop/create_training_set/satellite_training2/images/'

    output_path_satellite = 'C:/Users/Administrator/Desktop/create_training_set/texas_satellite_training2/lat_lon_sats.xlsx'
    #lat_lon_from_images(output_image_path, output_path_satellite)

    #create_images_from_excel(output_path,output_image_path)
    create_images_from_excel(manual_sat_latlons_path, output_image_path)