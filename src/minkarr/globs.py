import os.path as osp
import os


STORAGE_FOLDER = os.getenv('STORAGE_FOLDER')

if STORAGE_FOLDER is None:
    STORAGE_FOLDER = osp.join(osp.dirname(__file__), "../..")
else:
    STORAGE_FOLDER = osp.join(STORAGE_FOLDER, "minkarr_data")

if not STORAGE_FOLDER.endswith('/'):
    STORAGE_FOLDER += "/"
