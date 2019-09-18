
import requests
import pandas as pd

from IPython.display import Image


if __name__ == '__main__':

    url = '''https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom=20&size=1500x1500&maptype=satellite&markers=color:blue%7Clabel:S%7C40.702147,-74.015794&markers=color:green%7Clabel:G%7C40.711614,-74.012318&markers=color:red%7Clabel:C%7C40.718217,-73.998284&key=AIzaSyAnKtHsljkE0e8MRX2_fp55Ji8im0zjTLk'''
    a_url = url.format(latitude=29.178943,longitude=-97.055384)29.178943, -97.055384
    content = requests.get(a_url).content
    image = Image(content)

    print("done")