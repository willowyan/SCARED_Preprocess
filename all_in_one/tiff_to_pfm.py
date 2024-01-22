import numpy as np
import os
from PIL import Image
import sys
import cv2

def writePFM(file, image, scale=1):
    with open(file, 'wb') as file:
        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):  # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        file.write(('PF\n' if color else 'Pf\n').encode())
        file.write(f'{image.shape[1]} {image.shape[0]}\n'.encode())

        endian = image.dtype.byteorder

        if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
            scale = -scale

        file.write(f'{scale}\n'.encode())

        image.tofile(file)

def convert_tiff_to_pfm(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(input_folder_path):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            tiff_path = os.path.join(input_folder_path, filename)
            pfm_filename = os.path.splitext(filename)[0] + '.pfm'
            pfm_path = os.path.join(output_folder_path, pfm_filename)

            image = cv2.imread(tiff_path, -1)  # -1 to read the image as is
            if image is None:
                print(f"Failed to load {tiff_path}")
                continue

            image_np = np.array(image, dtype=np.float32)

            writePFM(pfm_path, image_np)
            print(f"Written PFM file: {pfm_path}")

input_folder = 'path_to_disparity_tiff'
output_folder = 'path_to_disparity_pfm'
convert_tiff_to_pfm(input_folder, output_folder)