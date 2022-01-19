#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:22:36 2021

@author: avanetten

Script to execute yoltv4 testing
"""

import pandas as pd
import skimage.io
import argparse
import shutil
import yaml
import time
import sys
import os


######################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

######################################
# 0. Load config and set variables
######################################

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    config_dict = yaml.safe_load(f)
    f.close()
config = dotdict(config_dict)
print("yoltv4_execute_test.py: config:")
print(config)

######################################
# 1. Import yoltv4 scripts
######################################

yolt_src_path = os.path.join(config.yoltv4_path, 'yoltv4')
print("yoltv4_execute_test.py: yolt_src_path:", yolt_src_path)
sys.path.append(yolt_src_path)
import prep_train
import tile_ims_labels
import post_process
import eval
import eval_errors


######################################
# 2. Prepare data
######################################
t0 = time.time()

###################
# object names
###################
# create name file
namefile = os.path.join(config.yoltv4_path, 'darknet', 'data', config.name_file_name)
for i, n in enumerate(config.object_names):
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
if config.sliceWidth > 0:
    # # make list of test files
    print("\nslicing im_dir:", config.test_im_dir)
    im_list = [z for z in os.listdir(config.test_im_dir) if z.endswith(config.im_ext)]
    if not os.path.exists(config.outdir_slice_ims):
        os.makedirs(config.outdir_slice_ims) #, exist_ok=True)
        os.makedirs(config.outdir_slice_txt) #, exist_ok=True)
        print("outdir_slice_ims:", config.outdir_slice_ims)
        # slice images
        for i,im_name in enumerate(im_list):
            im_path = os.path.join(config.test_im_dir, im_name)
            im_tmp = skimage.io.imread(im_path)
            h, w = im_tmp.shape[:2]
            print(i, "/", len(im_list), im_name, "h, w =", h, w)

            # tile data
            out_name = im_name.split('.')[0]
            tile_ims_labels.slice_im_plus_boxes(
                im_path, out_name, config.outdir_slice_ims,
                sliceHeight=config.sliceHeight, sliceWidth=config.sliceWidth,
                overlap=config.slice_overlap, slice_sep=config.slice_sep,
                skip_highly_overlapped_tiles=config.skip_highly_overlapped_tiles,
                overwrite=config.slice_overwrite,
                out_ext=config.out_ext, verbose=config.slice_verbose)
        im_list_test = []
        for f in sorted([z for z in os.listdir(config.outdir_slice_ims) if z.endswith(config.out_ext)]):
            im_list_test.append(os.path.join(config.outdir_slice_ims, f))
        df_tmp = pd.DataFrame({'image': im_list_test})
        df_tmp.to_csv(config.outpath_test_txt, header=False, index=False)
    else:
        print("Images already sliced to:", config.outdir_slice_ims)
        df_tmp = pd.read_csv(config.outpath_test_txt, names=['path'])
        im_list_test = list(df_tmp['path'].values)
else:
    # forego slicing
    im_list_test = []
    for f in sorted([z for z in os.listdir(config.test_im_dir) if z.endswith(config.im_ext)]):
        im_list_test.append(os.path.join(config.outdir_ims, f))
    df_tmp = pd.DataFrame({'image': im_list_test})
    df_tmp.to_csv(config.outpath_test_txt, header=False, index=False)
# print some values
print("N test images:", len(im_list))
print("N test slices:", len(df_tmp))
# view
print("head of test files ({})".format(config.outpath_test_txt))
with open(config.outpath_test_txt,'r') as f:
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
datafile = os.path.join(config.yoltv4_path, 'darknet', 'data', config.data_file_name)
os.system('echo classes = {} > {}'.format(len(config.object_names), datafile))
os.system('echo train = "nope" >> {}'.format(datafile))
os.system('echo valid = {} >> {}'.format(config.outpath_test_txt, datafile))
os.system('echo names = {} >> {}'.format(namefile, datafile))
os.system('echo backup = "nope" >> {}'.format(datafile))
# view
print("\ndata file ({})".format(datafile))
with open(datafile,'r') as f:
    all_lines = f.readlines()
    for i,l in enumerate(all_lines):
        print(l)


######################################
# 3. Execute GPU inference
######################################

cmd_dir = os.path.join(config.yoltv4_path, 'darknet')
# print("cmd_dir:", cmd_dir)
os.chdir(cmd_dir)
# print("os.getcwd():", os.getcwd())
yolt_cmd =  './darknet detector valid' \
            + ' ' + datafile \
            + ' ' + config.cfg_file \
            + ' ' + config.weights_file \
            + ' -out' + ' ' + config.outname_infer
# The following gives a segmentation fault every time!!!
# yolt_cmd =  os.path.join(config.yoltv4_path, 'darknet') + '/' + 'darknet' \
#             + ' detector valid' \
#             + ' ' + datafile \
#             + ' ' + config.cfg_file \
#             + ' ' + config.weights_file \
#             + ' -out' + ' ' + config.outname_infer
print("\nyolt_cmd:", yolt_cmd)
os.system(yolt_cmd)


######################################
# 4. Post process (CPU)
######################################

# strip off trailing '_' for outname_infer
outname = config.outname_infer[:-1]
print("post-proccessing:", outname)
for detection_thresh in config.detection_threshes:
    out_dir_root = os.path.join(config.yoltv4_path, 'darknet', 'results', outname)
    pred_dir = os.path.join(out_dir_root, 'orig_txt')
    out_csv = 'preds_refine_' + str(detection_thresh).replace('.', 'p') + '.csv'
    out_geojson_geo_dir = 'geojsons_geo_' + str(detection_thresh).replace('.', 'p')
    out_geojson_pix_dir = 'geojsons_pix_' + str(detection_thresh).replace('.', 'p')
    plot_dir = 'pred_plots_' + str(detection_thresh).replace('.', 'p')
    pred_txt_prefix = outname + '_'
    if config.extract_chips:
        out_dir_chips = 'detection_chips_' + str(detection_thresh).replace('.', 'p')
    else:
        out_dir_chips = ''

    # move raw predictions to output dir
    os.makedirs(os.path.join(out_dir_root, 'orig_txt'), exist_ok=True)
    mv_cmd = 'mv {}/darknet/results/{}*.txt {}/orig_txt/'.format(config.yoltv4_path, outname, out_dir_root)
    try:
        os.system(mv_cmd)
    except:
        pass
    # !mv {yoltv4_path}/darknet/results/{outname}*.txt {out_dir_root}/orig_txt/

    # post_process
    post_process.execute(
        pred_dir=pred_dir,
        truth_file=config.truth_file,
        raw_im_dir=config.test_im_dir,
        sliced_im_dir=config.outdir_slice_ims,
        out_dir_root=out_dir_root,
        out_csv=out_csv,
        ignore_names=config.ignore_names,
        out_geojson_geo_dir=out_geojson_geo_dir,
        out_geojson_pix_dir=out_geojson_pix_dir,
        plot_dir=plot_dir,
        im_ext=config.im_ext,
        out_dir_chips=out_dir_chips,
        chip_ext=config.chip_ext,
        chip_rescale_frac=config.chip_rescale_frac,
        allow_nested_detections=config.allow_nested_detections,
        max_edge_aspect_ratio=config.max_edge_aspect_ratio,
        nms_overlap_thresh=config.nms_overlap_thresh,
        slice_size=config.slice_size,
        sep=config.slice_sep,
        pred_txt_prefix=pred_txt_prefix,
        n_plots=config.n_plots,
        detection_thresh=detection_thresh)

tf = time.time()
print("\nTotal time to run inference and make plots:", tf - t0, "seconds")