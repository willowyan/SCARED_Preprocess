import cv2
import json
import os
from os.path import splitext, split
from os import listdir
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

"""
Read depth images and convert into disparity map.
"""

def tiff_reader(tiff_file):
    # Reads TIFF and splits it into two images along the vertical axis.

    raw = tiff.imread(tiff_file)
    img_l = raw[:1024,:,:] # First half
    img_r = raw[1024:,:,:] # Second half

    return img_l, img_r

def coor_to_disp(coor, Q):
    """
    Converts 3D coordinates to disparity using reprojection matrix Q.
    Returns: numpy.ndarray: Disparity map.    
    """
    # Parse the reprojection matrix
    Q = np.array(Q)
    fl = Q[2,3] # Focal length
    bl =  1 / Q[3,2] # Baseline
    cx = -Q[0,3] # Horizontal coord. of the principal point
    cy = -Q[1,3] # Vertical coord. of the principal point

    print('fl: ', fl, 'bl: ', bl, 'cx: ', cx, 'cy: ', cy)

    size = coor.shape[:2] # size[0] = 1024, size[1] = 1280
    X = coor[:,:,0]
    Y = coor[:,:,1]
    Z = coor[:,:,2]

    all_disp = np.zeros((size[0],size[1],2))
    disp = np.zeros(size)

    # Calculate disparity for each pixel
    for i in range(size[0]):
        for j in range(size[1]):
            x = X[i,j]
            y = Y[i,j]
            z = Z[i,j]

            if (z != 0):
                d = fl * bl / z # Disparity
                p_x = fl * x / z + cx # Projected x coord.
                p_y = fl * y / z + cy # Projected y coord.

                # Accumulate disparity for averaging
                if (p_x < size[1] and p_y < size[0]):
                    all_disp[int(p_y), int(p_x),0] += d
                    all_disp[int(p_y), int(p_x),1] += 1

                """
                if (p_x <= size[1] and p_y <= size[0]):
                    print('px: ', p_x, 'py: ', p_y, 'disp: ', d)
                    disp[int(p_y), int(p_x)] = d
                """
    for i in range(size[0]):
        for j in range(size[1]):
            if all_disp[i,j,1] != 0:
                disp[i,j] = all_disp[i,j,0] / all_disp[i,j,1]

    #print(disp.max(), disp.min())
    #plt.imshow(disp)
    #plt.show()
    return disp

def read_Q(reprojection_file):
    """
    Reads a reprojection matrix from a JSON file.
    Returns: list: Reprojection matrix Q
    """
    with open(reprojection_file) as json_file:
        data = json.load(json_file)
        Q = data['reprojection-matrix']

        return  Q


# def depth_to_disparity(path):
#     """
#     Processes all keyframes to convert depth map to disparity.
#     """
#     rootpath = path
#     keyframe_list = [os.path.join(rootpath, kf) for kf in listdir(rootpath) if ('keyframe' in kf and 'ignore' not in kf)]
    
#     for kf in keyframe_list:
#         # Paths for coord., reprojection data, and disparity maps
#         coor_filepath = os.path.join(rootpath,kf) + '/data/scene_points'
#         if not os.path.isdir(coor_filepath):
#             continue
#         reprojection_filepath = os.path.join(rootpath, kf) + '/data/reprojection_data'
#         disp_filepath = os.path.join(rootpath,kf) + '/data/disparity'

#         if not os.path.isdir(disp_filepath):
#             os.mkdir(disp_filepath)

#         frame_list = listdir(reprojection_filepath)

#         # Process each frame
#         for i in range(len(frame_list)):
#             reprojection_data = reprojection_filepath + '/frame_data%.6d.json' % i
#             coor_data = coor_filepath + '/scene_points%.6d.tiff' % i
#             disp_data = disp_filepath + '/frame_data%.6d.tiff' % i
#             print('Saving disparity to:', disp_data)

#             # Read reprojection matrix and TIFF file and calculate disparity
#             Q = read_Q(reprojection_data)
#             img_l, img_r = tiff_reader(coor_data)
#             disp = coor_to_disp(img_l, Q)
#             cv2.imwrite(disp_data, disp)

def write_pfm(file, image, scale=1):
    """
    Writes a numpy image to a PFM file.
    """
    with open(file, 'wb') as f:
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)  # PFM files store the image flipped vertically

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # grayscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        f.write(b'PF\n' if color else b'Pf\n')
        f.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        f.write(b'%f\n' % scale)

        image.tofile(f)

def depth_to_disparity_both(path):
    """
    Processes all keyframes to convert depth map to disparity and saves them in PFM format.
    """
    rootpath = path
    keyframe_list = [os.path.join(rootpath, kf) for kf in os.listdir(rootpath) if ('keyframe' in kf and 'ignore' not in kf)]

    for kf in keyframe_list:
        # Paths for coord., reprojection data, and disparity maps
        coor_filepath = os.path.join(rootpath, kf) + '/data/scene_points'
        if not os.path.isdir(coor_filepath):
            continue
        reprojection_filepath = os.path.join(rootpath, kf) + '/data/reprojection_data'
        disp_filepath_l = os.path.join(rootpath, kf) + '/data/disparity_left'
        disp_filepath_r = os.path.join(rootpath, kf) + '/data/disparity_right'

        if not os.path.isdir(disp_filepath_l):
            os.mkdir(disp_filepath_l)

        if not os.path.isdir(disp_filepath_r):
            os.mkdir(disp_filepath_r)

        frame_list = os.listdir(reprojection_filepath)

        # Process each frame
        for i in range(len(frame_list)):
            reprojection_data = reprojection_filepath + '/frame_data%.6d.json' % i
            coor_data = coor_filepath + '/scene_points%.6d.tiff' % i
            disp_data_l = disp_filepath_l + '/frame_data%.6d.pfm' % i  # Save as PFM
            disp_data_r = disp_filepath_r + '/frame_data%.6d.pfm' % i  # Save as PFM

            # Read reprojection matrix and TIFF file and calculate disparity
            Q = read_Q(reprojection_data)
            img_l, img_r = tiff_reader(coor_data)

            disp_l = coor_to_disp(img_l, Q)
            disp_r = coor_to_disp(img_r, Q)

            write_pfm(disp_data_l, disp_l)  # Save using the PFM writer
            write_pfm(disp_data_r, disp_r)  # Save using the PFM writer
            print('Saving disparity to:', disp_data_l, 'and', disp_data_r)

if __name__ == '__main__':
    path = '/media/eikoloki/TOSHIBA EXT/MICCAI_SCARED/dataset2'
    depth_to_disparity(path)
