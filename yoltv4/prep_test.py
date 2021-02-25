#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 2020
@author: avanetten
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
import skimage
import skimage.io
import skimage.transform
import fiona
import random
import multiprocessing
import cv2
# import solaris.vector
import shapely
import matplotlib
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

import geopandas as gpd
from rtree.core import RTreeError
import shutil

import circle_tile_ims_labels
from circle_prep_train import map_wrapper, prep_one


###############################################################################
###############################################################################
def main():
    verbose = True
    sliceHeight, sliceWidth = 416, 416
    slice_overlap=0.2
    n_threads = 8

    # data directory 
    data_dir = '/data/'
    data_dir_test = os.path.join(data_dir, 'test')

    # output dirs
    out_dir_root = '/wdata'
    # make dirs
    for d in [out_dir_root]:
        os.makedirs(d, exist_ok=True)

    # iterate through data, create masks, pngs, and labels
    subdirs = sorted(os.listdir(data_dir_test))
    shape_list = []
    input_args = []
    for i,subdir in enumerate(subdirs):
        print("\n")
        print(i, "/", len(subdirs), subdir)
        suff = ''
        pop = 'test'

        ##################
        # Read in data
        ##################

        dir_tot = os.path.join(data_dir_test, subdir)
        mul_path = os.path.join(dir_tot, subdir + '_MUL.tif')
        pan_path = os.path.join(dir_tot, subdir + '_PAN.tif')
        # set pan-sharpen folder
        out_dir_tiff = os.path.join(out_dir_root, 'PS-RGB_test')
        os.makedirs(out_dir_tiff, exist_ok=True)
        ps_rgb_path = os.path.join(out_dir_tiff, subdir + '_PS-RGB.tif')
                 
        # image 
        im_tmp = skimage.io.imread(pan_path)
        h, w = im_tmp.shape[:2]
        shape_list.append([subdir + '_PS-RGB', h, w])
        # im_tmp = skimage.io.imread(pan_path)
        # h, w = im_tmp.shape
        # shape_list.append([subdir + '_PAN', h, w])
        aspect_ratio = 1.0 * h / w
        dx = np.abs(h - w)
        max_dx = 3
 
        if verbose:
            print("  h, w:", h, w)
            print("  aspect_ratio:", aspect_ratio)
                   
        ##################
        # Set output paths
        ##################
        
        # check if it's a huge square image (these all have a large circle centered in the middle,
        #  so we can skip for training)
        if (((h >= 600) and (w >= 600) and (dx <= max_dx)) \
            or ((h >= 1000) and (w >= 1000) and (0.97 < aspect_ratio < 1.03))):
            # or (h * w > 800 * 800):  # original version (no tiling)
            suff = '_yuge'
        
        # look for large images with multiple annotations
        elif ((h >= 580) and (w >= 580)): 
            # and (len(annotations) > 1):        
            suff = '_tile'
        
        else:
            suff = ''
            
        # set output folders
        out_dir_image = os.path.join(out_dir_root, pop, 'images' + suff)
        for d in [out_dir_image]:
            os.makedirs(d, exist_ok=True)
            
        # output files
        out_path_image = os.path.join(out_dir_image, subdir + '_PS-RGB.png')
        # out_path_image = os.path.join(out_dir_image, subdir + '_PAN.png')
        
        if not os.path.exists(out_path_image):
            input_args.append([prep_one,
                    pan_path, mul_path, None, ps_rgb_path, 
                    subdir, suff, 
                    out_dir_image, None, None,
                    out_path_image, None, None,
                    sliceHeight, sliceWidth, 255, 
                    0,
                    verbose])

    ##################
    # Execute
    ##################
    print("len input_args", len(input_args))
    print("Execute...\n")
    with multiprocessing.Pool(n_threads) as pool:
        pool.map(map_wrapper, input_args)


        # ##################
        # # Process data
        # ##################
        #
        #
        # # tile data, if needed
        # if suff == '_tile':
        #     # tile image, labels, and mask
        #     out_name = subdir + '_PS-RGB'
        #     # out_name = subdir + '_PAN'
        #     # tile (also creates labels)
        #     circle_tile_ims_labels.slice_im_plus_boxes(
        #         ps_rgb_path, out_name, out_dir_image,
        #         boxes=[], yolo_classes=[], out_dir_labels=None,
        #         mask_path=None, out_dir_masks=None,
        #         sliceHeight=sliceHeight, sliceWidth=sliceWidth,
        #         overlap=slice_overlap, slice_sep='|',
        #         out_ext='.png', verbose=False)
        #
        # # no tiling
        # else:
        #
        #     # simply copy to dest folder if object is yuge
        #     if suff == '_yuge':
        #         shutil.copyfile(ps_rgb_path, out_path_image)
        #         hfinal, wfinal = h, w
        #
        #     # simply copy to dest folder if aspect ratio is reasonable
        #     elif (0.9 < aspect_ratio < 1.1):
        #         shutil.copyfile(ps_rgb_path, out_path_image)
        #         hfinal, wfinal = h, w
        #
        #     # else let's add a border on right or bottom,
        #     #  (which doesn't affect pixel coords of labels).
        #     else:
        #         topBorderWidth, bottomBorderWidth, leftBorderWidth, rightBorderWidth = 0, 0, 0, 0
        #         if h / w > 1.1:
        #             rightBorderWidth = np.abs(h - w)
        #         if h / w < 0.9:
        #             bottomBorderWidth = np.abs(h - w)
        #         # add border to image
        #         # im_tmp = cv2.imread(out_path_image, 1) # make everything 3-channel?
        #         outputImage = cv2.copyMakeBorder(
        #                      im_tmp,
        #                      topBorderWidth,
        #                      bottomBorderWidth,
        #                      leftBorderWidth,
        #                      rightBorderWidth,
        #                      cv2.BORDER_CONSTANT,
        #                      value=0)
        #         skimage.io.imsave(out_path_image, outputImage)
        #         # cv2.imwrite(out_path_image, outputImage)
    
    # save shapes
    outpath_shapes = os.path.join(out_dir_root, 'shapes_test.csv')  
    df_shapes = pd.DataFrame(shape_list, columns=['im_name', 'h', 'w'])
    df_shapes.to_csv(outpath_shapes)

    #################
    # lists of images
        
    # normal test images
    dtmp = os.path.join(out_dir_root, 'test', 'images' + '')
    im_list_test = []
    for f in sorted([z for z in os.listdir(dtmp) if z.endswith('.png')]):
        im_list_test.append(os.path.join(dtmp, f))
    # tile test images
    dtmp = os.path.join(out_dir_root, 'test', 'images' + '_tile')
    im_list_test_tile = []
    for f in sorted([z for z in os.listdir(dtmp) if z.endswith('.png')]):
        im_list_test_tile.append(os.path.join(dtmp, f))
    # yuge test images
    dtmp = os.path.join(out_dir_root, 'test', 'images' + '_yuge')
    im_list_test_yuge = []
    for f in sorted([z for z in os.listdir(dtmp) if z.endswith('.png')]):
        im_list_test_yuge.append(os.path.join(dtmp, f))

    # combine in different csvs    
    outpath_tmp = os.path.join(out_dir_root, 'test.txt')
    list_tmp = im_list_test
    df_tmp = pd.DataFrame({'image': list_tmp})
    df_tmp.to_csv(outpath_tmp, header=False, index=False)
    # test + tile
    outpath_tmp = os.path.join(out_dir_root, 'test+tile.txt')
    list_tmp = im_list_test + im_list_test_tile
    df_tmp = pd.DataFrame({'image': list_tmp})
    df_tmp.to_csv(outpath_tmp, header=False, index=False)
    # test + tile + yuge
    outpath_tmp = os.path.join(out_dir_root, 'test+tile+yuge.txt')
    list_tmp = im_list_test + im_list_test_tile + im_list_test_yuge
    df_tmp = pd.DataFrame({'image': list_tmp})
    df_tmp.to_csv(outpath_tmp, header=False, index=False)
    

###############################################################################
###############################################################################
if __name__ == "__main__":
    main()