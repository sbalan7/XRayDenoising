from PIL import Image, ImageOps
import cv2

import numpy as np
import os


def resize_with_padding(img, expected_size):
    img = Image.fromarray(img)
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return np.asarray(ImageOps.expand(img, padding))

filenames = os.listdir('Images/')

for filename in filenames:
    if filename.endswith('.tif'):
        I = cv2.imread('Images/'+filename)
        print(f'{filename[:-4].replace("_", " ")}, Shape = {I.shape} and Data = {I.dtype}')
        I = resize_with_padding(I, (max(I.shape), max(I.shape)))
        print(f'After padding, Shape = {I.shape} and Data = {I.dtype}')
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        print(f'After grayscale, Shape = {I.shape} and Data = {I.dtype}')
        I = cv2.resize(I, (1000, 1000))
        print(f'After resize, Shape = {I.shape} and Data = {I.dtype}')
        cv2.imwrite('Images/proc_'+filename, I)




