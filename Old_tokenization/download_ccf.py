"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python download_ccf.py
Input: None (Downloads from Allen API)
Output: annotation_25.nrrd file containing the CCFv3 3D volume
"""

import urllib.request
import os

url = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd"
filename = "annotation_25.nrrd"

if not os.path.exists(filename):
    print(f"Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete!")
else:
    print(f"{filename} already exists.")
