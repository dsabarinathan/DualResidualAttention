
from keras.models import load_model

from numpy import newaxis
import numpy as np
import cv2
import os
import argparse
import time
from coord import CoordinateChannel2D
from model_utils import sum_squared_error, ssim,PSNR

import hdf5storage


from Conf import RGB_FILTER_CSV, JPEG_QUALITY, MOSAIC_FILTER_CSV, CROP, SUBMISSION_SIZE_LIMIT
from NTIRE2022Util import load_rgb_filter, createNoisyRGB, save_jpg, loadCube, create_multispectral, load_ms_filter


def saveCube(path, cube, bands=None, norm_factor=None):
    """
    Save a spectra cube in Matlab HDF5 format
    :param path: Destination filename as full path
    :param cube: Spectral cube as Numpy array
    :param bands: Bands of spectral cube as Numpy array
    :param norm_factor: Normalization factor to source image counts
    """
    hdf5storage.write({u'cube': cube,
                       u'bands': bands,
                       u'norm_factor': norm_factor}, '.',
                       path, matlab_compatible=True)



def copy_patch1(x, y):
    x[:] = y[:]


def copy_patch2(stride, h, x, y):

    x[:,:, :-(h % stride),:] = (y[:, :, :-(h % stride),:] + x[:, :, :-(h % stride),:]) / 2.0
    x[:, :, -(h % stride):,:] = y[:, :, -(h % stride):, :]


def copy_patch3(stride, w, x, y):
    x[:, :-(w % stride), :,:] = (y[:, :-(w % stride),:,:] + x[:, :-(w % stride),:,:]) / 2.0
    x[:,-(w % stride):, :,:] = y[:, -(w % stride):, :, :]


def copy_patch4(stride, w, h, x, y):
    x[:,:-(w % stride),:,:] = (y[:, :-(w % stride),:,:] + x[:, :-(w % stride), :, :]) / 2.0
    x[:,-(w % stride):, :-(h % stride),:] = (y[:, -(w % stride):, :-(h % stride),:] + x[:, -(w % stride):, :-(h % stride),:]) /2.0
    x[:,-(w % stride):, -(h % stride):,:] = y[:, -(w % stride):, -(h % stride):,:]


def reconstruction_patch_image_gpu(imr0, model, patch, stride):
    all_time = 0
    rgb = np.expand_dims(imr0, axis=0).copy()
    _, w, h ,c= rgb.shape
#    rgb = torch.from_numpy(rgb).float()
    temp_hyper =np.zeros((1,w,h,31)).astype(np.float32)

#    temp_hyper = torch.zeros(1, 31, w, h).float()
    # temp_rgb = torch.zeros(1, 3, w, h).float()
    for x in range(w//stride + 1):
        for y in range(h//stride + 1):
            if x < w // stride and y < h // stride:
                rgb_patch0 = rgb[:, x * stride:x * stride + patch, y * stride:y * stride + patch,:]
                hyper_patch = model.predict(rgb_patch0)

                copy_patch1(temp_hyper[:, x * stride:x * stride + patch, y * stride:y * stride + patch, :], hyper_patch)

            elif x < w // stride and y == h // stride:
                rgb_patch1 = rgb[:, x * stride:x * stride + patch, -patch:, :]
                hyper_patch = model.predict(rgb_patch1)

                copy_patch2(stride, h, temp_hyper[:, x * stride:x * stride + patch, -patch:,:], hyper_patch)
            elif x == w // stride and y < h // stride:
                rgb_patch2 = rgb[:, -patch:, y * stride:y * stride + patch,:]
                hyper_patch = model.predict(rgb_patch2)
                copy_patch3(stride, w, temp_hyper[:, -patch:, y * stride:y * stride + patch,:], hyper_patch)
            else:
                rgb_patch3 = rgb[:, -patch:, -patch:,:]
                hyper_patch = model.predict(rgb_patch3)
                copy_patch4(stride, w, h, temp_hyper[:, -patch:, -patch:, :], hyper_patch)
#            all_time += patch_time
            
    temp_hyper = np.float32(temp_hyper)

    
    return temp_hyper

if __name__ == '__main__':
         
    parser = argparse.ArgumentParser(description='eye-net')
    parser.add_argument("--testImagePath", type=str,dest="test_path" ,help="Path of test Images",default='./test/',action="store")
    args = parser.parse_args()
    

    model = load_model('./model/Model--HierarchicalNetwork-residual-16-24-4915.4316--val_mean_squared_error-0.0097.hdf5',custom_objects={'sum_squared_error':sum_squared_error,'ssim':ssim,'CoordinateChannel2D':CoordinateChannel2D,'PSNR':PSNR})

    output_path = './output_file_model/'
    if not os.path.exists(output_path):
       os.makedirs(output_path)
    
    output_path_cropped = './output_path_model_cropped/'
    if not os.path.exists(output_path_cropped):
       os.makedirs(output_path_cropped)
       
    testImagePath = args.test_path
    
    fileName = os.listdir(testImagePath)
    
    for i in range(len(fileName)):
        
        start_time = time.time()

        img = cv2.imread(testImagePath+fileName[i])
   
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        outputNew=reconstruction_patch_image_gpu(img/255, model,64, 64)
        
        end_time = time.time()
    
        print('predicted time', end_time-start_time)
        print(fileName[i].split('clean')[0][0:-1]+'.mat')

        saveCube(output_path+fileName[i][0:-4]+'.mat',cube=outputNew[0])
        cube, bands = loadCube(output_path+fileName[i][0:-4]+'.mat')

        cube = cube[CROP]

        # Save cropped file
        saveCube(output_path_cropped+fileName[i][0:-4]+'.mat', cube, bands=bands)

#        print(i)
        
    print("output files saved in "+output_path_cropped)
