import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# LOAD FILES AND MAKE IT AS NUMPY FILE
import os
import glob

# TWO LEVELS UP
dir_path = os.path.normpath(os.path.join(__file__,'../../', 'Dataset'))
#dir_path = os.path.dirname(os.path.realpath(__file__))

patchlist = [
    glob.glob(os.path.join(dir_path, "Patch1k", "Hair", "Training", "*.jpg")),
    glob.glob(os.path.join(dir_path, "Patch1k", "Hair", "Test", "*.jpg")),
    glob.glob(os.path.join(dir_path, "Patch1k", "NonHair", "Training", "*.jpg")),
    glob.glob(os.path.join(dir_path, "Patch1k", "NonHair", "Test", "*.jpg"))
]

# RESIZE IMAGES
import cv2
width = 35
height = 35

def resizing_images(input) :
    x = []
    for img in input:
        base = os.path.basename(img)
        # Read and resize image
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (width,height), interpolation=cv2.INTER_CUBIC))
    return x

hair_train = resizing_images(patchlist[0])
hair_test = resizing_images(patchlist[1])
nonhair_train = resizing_images(patchlist[2])
nonhair_test = resizing_images(patchlist[3])

np.savez(os.path.normpath(os.path.join(__file__,'../', 'Patch1k_resized_35.npz')), hair_train=hair_train, hair_test=hair_test, nonhair_train=nonhair_train, nonhair_test=nonhair_test)

with np.load(os.path.normpath(os.path.join(__file__,'../', 'Patch1k_resized.npz'))) as fh :
    print(fh.files)
