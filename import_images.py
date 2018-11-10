import numpy as np
from PIL import Image

# LOAD FILES AND MAKE IT AS NUMPY FILE
import os
import glob

# TWO LEVELS UP
dir_path = os.path.normpath(os.path.join(__file__,'../../', 'Dataset'))
#dir_path = os.path.dirname(os.path.realpath(__file__))

filelist = [
    glob.glob(os.path.join(dir_path, "Figaro-1k", "Original", "Training", "*.jpg")),
    glob.glob(os.path.join(dir_path, "Figaro-1k", "Original", "Test", "*.jpg")),
    glob.glob(os.path.join(dir_path, "Figaro-1k", "GT", "Training", "*.pbm")),
    glob.glob(os.path.join(dir_path, "Figaro-1k", "GT", "Test", "*.pbm"))
]

x_train = np.array([np.array(Image.open(fname)) for fname in filelist[0]])
x_test = np.array([np.array(Image.open(fname)) for fname in filelist[1]])
y_train = np.array([np.array(Image.open(fname)) for fname in filelist[2]])
y_test = np.array([np.array(Image.open(fname)) for fname in filelist[3]])

# CHECK THE SHAPE OF EACH ARRAY, EACH IMAGE
print(x_train.shape)
print(x_train[0].shape)
print(x_test.shape)
print(x_test[0].shape)
print(y_train.shape)
print(y_train[0].shape)
print(y_test.shape)
print(y_test[0].shape)
