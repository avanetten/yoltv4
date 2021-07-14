#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:22:36 2021

@author: avanetten

Script to execute yoltv4 testing
"""


import skimage.io
import shutil
import time
import sys
import os

######################################
# 0. Set variables
######################################

###################
# yoltv4 input file variables
yoltv4_path =  '/opt/yoltv4/'
# object name variables (object names should be in order)
name_file_name = 'yoltv2_5class.name'
object_names = ['small_aircraft', 'large_aircraft', 'small_vehicle', 'bus_truck', 'boat']
# data file variable
data_file_name = 'yoltv2_5class_test.data'

###################
# image slicing variables
data_root = 'test_images/'
im_dir =  os.path.join(data_root, 'raw')
sliceHeight, sliceWidth = 544, 544
slice_overlap = 0.2
im_ext = '.png'
out_ext = '.jpg'
# shouldn't need changed below here
skip_highly_overlapped_tiles=False
slice_verbose = False
n_threads_slice = 8
slice_sep = '__'
slice_overwrite = False
outdir_slice_root = os.path.join(data_root, 'yoltv4')
outdir_slice_ims = os.path.join(outdir_slice_root, 'images_slice')
outdir_slice_txt = os.path.join(outdir_slice_root, 'txt')
outpath_test_txt = os.path.join(outdir_slice_txt, 'test.txt')


###################
# inference variables
outname_infer = 'yoltv2_5class_'
cfg_file = 'ave_dense_544.cfg' 
weights_file = 'weights_5class_vehicles/ave_dense_544_final.weights'

###################
# post-process variables
detection_threshes = [0.2] 
n_plots = 8
allow_nested_detections = True
truth_file = '' # os.path.join(data_root, 'test', 'geojsons_pix_comb', truth_file_name)
# seldom changed below here
extract_chips = False
chip_rescale_frac = 1.1
chip_ext = '.jpg'
slice_size = sliceWidth

###################
# import yoltv4 scripts
sys.path.append(os.path.join(yoltv4_path, 'yoltv4'))
import prep_train
import tile_ims_labels
import post_process
import eval
import eval_errors


######################################
# 1. Prepare data
######################################
t0 = time.time()

###################
# object names
###################
# create name file
namefile = os.path.join(yoltv4_path, 'darknet', 'data', name_file_name)
for i, n in enumerate(object_names):
    if i == 0:
        os.system( 'echo {} > {}'.format(n, namefile))
    else:
        os.system( 'echo {} >> {}'.format(n, namefile))
# view
print("\nobject names ({})".format(namefile))
with open(namefile,'r') as f:
    all_lines = f.readlines()
    for l in all_lines:
        print(l)

###################
# slice test images
###################
if sliceWidth > 0:
    print("\nslicing im_dir:", im_dir)
    im_list = [z for z in os.listdir(im_dir) if z.endswith(im_ext)]
    if not os.path.exists(outdir_slice_ims):
        os.makedirs(outdir_slice_ims) #, exist_ok=True)
        os.makedirs(outdir_slice_txt) #, exist_ok=True)
        print("outdir_slice_ims:", outdir_slice_ims)
        # slice images
        for i,im_name in enumerate(im_list):
            im_path = os.path.join(im_dir, im_name)
            im_tmp = skimage.io.imread(im_path)
            h, w = im_tmp.shape[:2]
            print(i, "/", len(im_list), im_name, "h, w =", h, w)

            # tile data
            out_name = im_name.split('.')[0]
            tile_ims_labels.slice_im_plus_boxes(
                im_path, out_name, outdir_slice_ims,
                sliceHeight=sliceHeight, sliceWidth=sliceWidth,
                overlap=slice_overlap, slice_sep=slice_sep,
                skip_highly_overlapped_tiles=skip_highly_overlapped_tiles,
                overwrite=slice_overwrite,
                out_ext=out_ext, verbose=slice_verbose)
        # make list of test files
        im_list_test = []
        for f in sorted([z for z in os.listdir(outdir_slice_ims) if z.endswith(out_ext)]):
            im_list_test.append(os.path.join(outdir_slice_ims, f))
        df_tmp = pd.DataFrame({'image': im_list_test})
        df_tmp.to_csv(outpath_test_txt, header=False, index=False)
    else:
        print("images already sliced to:", outdir_slice_ims)
else:
    # forego slicing
    im_list_test = []
    for f in sorted([z for z in os.listdir(im_dir) if z.endswith(im_ext)]):
        im_list_test.append(os.path.join(outdir_ims, f))
    df_tmp = pd.DataFrame({'image': im_list_test})
    df_tmp.to_csv(outpath_test_txt, header=False, index=False)
# print some values
print("N test images:", len(im_list_test))
print("N test slices:", len(df_tmp))
# view
print("head of test files ({})".format(outpath_test_txt))
with open(outpath_test_txt,'r') as f:
    all_lines = f.readlines()
    for i,l in enumerate(all_lines):
        if i < 5:
            print(l)
        else:
            break
            
###################
# make data file
###################
# create file
datafile = os.path.join(yoltv4_path, 'darknet', 'data', data_file_name)
os.system('echo classes = {} > {}'.format(len(object_names), datafile))
os.system('echo train = "nope" >> {}'.format(datafile))
os.system('echo valid = {} >> {}'.format(outpath_test_txt, datafile))
os.system('echo names = {} >> {}'.format(namefile, datafile))
os.system('echo backup = backup/ >> {}'.format(datafile))
# view
print("\ndata file ({})".format(datafile))
with open(datafile,'r') as f:
    all_lines = f.readlines()
    for i,l in enumerate(all_lines):
        print(l)

######################################
# 2. Execute 
######################################

###################
# Run infernence 
###################
os.system('cd {}'.format(os.path.join(yoltv4_path, 'darknet')))
yolt_cmd =  os.path.join(yoltv4_path, 'darknet') + '/' + 'darknet' \
            + ' detector valid' \
            + ' ' + datafile \
            + ' ' + cfg_file \
            + ' ' + weights_file \
            + ' -out' + ' ' + outname_infer
print("\nyolt_cmd:", yolt_cmd)
os.system(yolt_cmd)

###################
# Post process
###################

# strip off trailing '_' for outname_infer
outname = outname_infer[:-1]
print("post-proccessing:", outname)
for detection_thresh in detection_threshes:
    out_dir_root = os.path.join(yoltv4_path, 'darknet', 'results', outname)
    pred_dir = os.path.join(out_dir_root, 'orig_txt')
    out_csv = 'preds_refine_' + str(detection_thresh).replace('.', 'p') + '.csv'
    out_geojson_geo_dir = 'geojsons_geo_' + str(detection_thresh).replace('.', 'p')
    out_geojson_pix_dir = 'geojsons_pix_' + str(detection_thresh).replace('.', 'p')
    plot_dir = 'pred_plots_' + str(detection_thresh).replace('.', 'p')
    pred_txt_prefix = outname + '_'
    if extract_chips:
        out_dir_chips = 'detection_chips_' + str(detection_thresh).replace('.', 'p')
    else:
        out_dir_chips = ''

    # move raw predictions to output dir
    os.makedirs(os.path.join(out_dir_root, 'orig_txt'), exist_ok=True)
    mv_cmd = 'mv {}/darknet/results/{}*.txt {}/orig_txt/'.format(yoltv4_path, outname, out_dir_root)
    try:
        os.system(mv_cmd)
    except:
        pass
    # !mv {yoltv4_path}/darknet/results/{outname}*.txt {out_dir_root}/orig_txt/

    # post_process
    post_process.execute(
        pred_dir=pred_dir,
        truth_file=truth_file,
        raw_im_dir=im_dir,
        sliced_im_dir=outdir_slice_ims,
        out_dir_root=out_dir_root,
        out_csv=out_csv,
        out_geojson_geo_dir=out_geojson_geo_dir,
        out_geojson_pix_dir=out_geojson_pix_dir,
        plot_dir=plot_dir,
        im_ext=im_ext,
        out_dir_chips=out_dir_chips,
        chip_ext=chip_ext,
        chip_rescale_frac=chip_rescale_frac,
        allow_nested_detections=allow_nested_detections,
        slice_size=slice_size,
        sep=slice_sep,
        pred_txt_prefix=pred_txt_prefix,
        n_plots=n_plots,
        detection_thresh=detection_thresh)

tf = time.time()
print("\nTotal time to run inference and make plots:", tf - t0, "seconds")