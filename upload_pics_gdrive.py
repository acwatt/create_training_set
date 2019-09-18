from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
# Importing os and glob to find all PDFs inside subfolder
import glob, os
from datetime import date


def upload_folder(folder_path):
    # Login to Google Drive and create drive object
    print(os.getcwd())
    g_login = GoogleAuth()
    g_login.LocalWebserverAuth()
    drive = GoogleDrive(g_login)

    today = date.today()
    # YY/mm/dd
    today = today.strftime("%Y/%m/%d")

    os.chdir(folder_path)
    for file in glob.glob("*.jpg"):
        print(file)
        with open(file, "r") as f:
            fn = os.path.basename(f.name)
            file_drive = drive.CreateFile({'title': fn})
        file_drive.SetContentString(f.read())
        file_drive.Upload()
        print("The file: " + fn + " has been uploaded")

    print("All files have been uploaded")

if __name__=="__main__":
    ## Google drive info
    API_KEYID = 'AIzaSyBhe7e9ixkWrPGIoyBfVuNf83SjFZeYZPM'
    folder_to_upload = 'C:/Users/Administrator/Desktop/create_training_set/drone_training2/images/outsample/'
    upload_folder(folder_to_upload)

    client_ID = '684635950141-5tkdgefmnbt8mkd8hhgbpu8mlennpavh.apps.googleusercontent.com'
    client_secret = '8jMgNr0IVY3anYU85Vb4HDYh'