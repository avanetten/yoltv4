"""
@author: avanetten

Evaluate error bars of predictions

Adapted from https://github.com/avanetten/simrdwn/blob/master/simrdwn/core/simrdwn_eval_errors.py

"""

from functools import partial
import multiprocessing
import pandas as pd
import scipy.stats
import numpy as np
import random
import glob
import os

import eval


###############################################################################
def map_wrapper(x):
    '''For multi-threading'''
    return x[0](*(x[1:]))
    

###############################################################################
def _bootstrap_holding_func(proposal_polygons_dir, gt_polygons_dir,
                  geojson_names_subset=[],
                  prediction_cat_attrib="class", gt_cat_attrib='make',
                  object_subset=[], threshold=0.5, 
                  confidence_attrib="confidence",
                  file_format="geojson",
                  verbose=False,
                  iteration=0):
    '''For multi-threading bootstrapping.
       Output is a list of form:
            [iter, object_classes, df]'''
                  
    if verbose:
        outstr = "\nIteration=" + str(iteration)
        print(outstr)
        
    # get performance
    object_classes, mAP, APs_by_class, mF1_score, f1s_by_class, \
        precision_iou_by_obj, precision_by_class, mPrecision, \
        recall_iou_by_obj, recall_by_class, mRecall, confidences = \
                eval.mAP_score(proposal_polygons_dir, gt_polygons_dir,
                              geojson_names_subset=geojson_names_subset,
                              prediction_cat_attrib=prediction_cat_attrib, 
                              gt_cat_attrib=gt_cat_attrib,
                              object_subset=object_subset, 
                              threshold=threshold, 
                              confidence_attrib=confidence_attrib,
                              file_format=file_format)
    
    # ensure classes don't change
    if object_classes != object_subset:
        print("Object classes don't match!")
        return

    # simple output
    boot_list_raw = [iteration, mAP, mF1_score, object_classes, APs_by_class, f1s_by_class]
    # return boot_list
    
    # refine output to create a dataframe
    score_dict = {'mAP': mAP, 'mF1': mF1_score}
    for (ob, ob_ap, ob_f1) in zip(object_classes, APs_by_class, f1s_by_class):
        score_dict[ob + '_AP'] = ob_ap
        score_dict[ob + '_F1'] = ob_f1
    df = pd.DataFrame(score_dict, index=[iteration])

    return df  #, boot_list_raw  
     
        
###############################################################################
def mAP_bootstrap(proposal_polygons_dir, gt_polygons_dir,
                  geojson_names_subset=[],
                  prediction_cat_attrib="class", gt_cat_attrib='make',
                  object_subset=[], threshold=0.5, 
                  confidence_attrib="confidence",
                  file_format="geojson",
                  N_bootstraps=500,
                  N_threads=10,
                  verbose=False):
    """ 
    Bootstrap F1 and mAP error from output of eval.py (multi-threaded).
    Assume there are multiple observations, and sample with replacement 
    from the test images to compute error bars.
    Use mAP_score() from eval.py

    Arguments
    ---------
        proposal_polygons_dir : str
            The path that contains any model proposal polygons
        gt_polygons_dir : str
            The path that contains the ground truth polygons
        geojson_names_subset: list
            List of geojson names to consider (if [], use all geojsons)
        prediction_cat_attrib : str
            The column or attribute within the predictions that specifies
            unique classes
        gt_cat_attrib : str
            The column or attribute within the ground truth that
            specifies unique classes
        object_subset : list
            A list or subset of the unique objects that are contained within the
            proposal and ground truth polygons. If empty, this will be
            auto-created using all classes that appear in the proposal and
            ground truth polygons.
        threshold : float
            A value between 0.0 and 1.0 that determines the IOU threshold for a
            true positve.
        confidence_attrib : str
            The column or attribute within the proposal polygons that
            specifies model confidence for each prediction
        file_format : str
            The extension or file format for predictions
        N_bootstraps : int
            Number of bootstraps
        N_threads : int
            Number of threads for multiprocessing
        verbose : bool
            Verbose switch
                  
    Returns
    ---------
        object_subset : list
            All unique objects that exist in the ground truth polygons
        mAP : float
            The mean average precision score of APs_by_class
        APs_by_class : list
            A list containing the AP score for each class
        mF1 : float
            The mean F1 score of f1s_by_class
        f1s_by_class : list
            A list containing the f1 score for each class
        precision_iou_by_obj : list of lists
            An iou score for each object per class (precision specific)
        precision_by_class : list
            A list containing the precision score for each class
        mPrecision : float
            The mean precision score of precision_by_class
        recall_iou_by_obj : list of lists
            An iou score for each object per class (recall specific)
        recall_by_class : list
            A list containing the recall score for each class
        mRecall : float
            The mean recall score of recall_by_class
        confidences : list of lists
            All confidences for each object for each class
    """

    # get geojson names
    if len(geojson_names_subset) > 0:
        gt_geojsons = geojson_names_subset
    else:
        os.chdir(gt_polygons_dir)
        search = "*" + file_format
        gt_geojsons = glob.glob(search)
    len_bootstrap_sample = len(gt_geojsons)
    
    # get objects
    if len(object_subset) == 0:
        if verbose:
            print("getting unique objects...")
        prop_objs, object_subset, all_objs = eval.get_all_objects(
            proposal_polygons_dir, gt_polygons_dir,
            geojson_names_subset=[],
            prediction_cat_attrib=prediction_cat_attrib,
            gt_cat_attrib=gt_cat_attrib, file_format=file_format)
    if verbose:
        print("N objects = ", len(object_subset), "objects:", object_subset)
    
    # populate out_dict
    
    out_dict = {'mAP': [], 'mF1': []}
    for ob in object_subset:
        out_dict[ob] = {'AP':[], 'F1': []}
        
    input_args = []
    for i in range(N_bootstraps):
        # sample with replacement
        geojson_names_boot = random.choices(gt_geojsons, k=len_bootstrap_sample)
        # if verbose:
        #     print("\nBootstrap:", i, "/", N_bootstraps)
        #     print("  geojsons_boot[:4]:", geojson_names_boot[:4])
        input_args.append([_bootstrap_holding_func,
            proposal_polygons_dir, gt_polygons_dir,
            geojson_names_boot,
            prediction_cat_attrib,
            gt_cat_attrib,
            object_subset,
            threshold, 
            confidence_attrib,
            file_format,
            verbose,
            i])

    print("Execute...\n")
    pool = multiprocessing.Pool(N_threads)
    df_list = pool.map(map_wrapper, input_args)
    
    # build total dataframe
    df_tot = pd.concat(df_list)
    
    # compute means and stds 
    # (make sure to compute prior to updating dataframe)
    means = df_tot.mean(axis=0)
    stds = df_tot.std(axis=0)
    df_tot.loc['mean'] = means
    df_tot.loc['std'] = stds
    
    # return object_subset, nested_outputs
    return object_subset, df_tot
    

###############################################################################
def mAP_bootstrap_single_thread(proposal_polygons_dir, gt_polygons_dir,
                  geojson_names_subset=[],
                  prediction_cat_attrib="class", gt_cat_attrib='make',
                  object_subset=[], threshold=0.5, 
                  confidence_attrib="confidence",
                  file_format="geojson",
                  N_bootstraps=500,
                  verbose=False):
    """ 
    Bootstrap F1 and mAP error from output of eval.py.
    Single-threaded and quite slow. 
    Assume there are multiple observations, and sample with replacement 
    from the test images to compute error bars.
    Use mAP_score() from eval.py

    Arguments
    ---------
        proposal_polygons_dir : str
            The path that contains any model proposal polygons
        gt_polygons_dir : str
            The path that contains the ground truth polygons
        geojson_names_subset: list
            List of geojson names to consider (if [], use all geojsons)
        prediction_cat_attrib : str
            The column or attribute within the predictions that specifies
            unique classes
        gt_cat_attrib : str
            The column or attribute within the ground truth that
            specifies unique classes
        object_subset : list
            A list or subset of the unique objects that are contained within the
            proposal and ground truth polygons. If empty, this will be
            auto-created using all classes that appear in the proposal and
            ground truth polygons.
        threshold : float
            A value between 0.0 and 1.0 that determines the IOU threshold for a
            true positve.
        confidence_attrib : str
            The column or attribute within the proposal polygons that
            specifies model confidence for each prediction
        file_format : str
            The extension or file format for predictions
    Returns
    ---------
        object_subset : list
            All unique objects that exist in the ground truth polygons
        mAP : float
            The mean average precision score of APs_by_class
        APs_by_class : list
            A list containing the AP score for each class
        mF1 : float
            The mean F1 score of f1s_by_class
        f1s_by_class : list
            A list containing the f1 score for each class
        precision_iou_by_obj : list of lists
            An iou score for each object per class (precision specific)
        precision_by_class : list
            A list containing the precision score for each class
        mPrecision : float
            The mean precision score of precision_by_class
        recall_iou_by_obj : list of lists
            An iou score for each object per class (recall specific)
        recall_by_class : list
            A list containing the recall score for each class
        mRecall : float
            The mean recall score of recall_by_class
        confidences : list of lists
            All confidences for each object for each class
    """

    # get geojson names
    if len(geojson_names_subset) > 0:
        gt_geojsons = geojson_names_subset
    else:
        os.chdir(gt_polygons_dir)
        search = "*" + file_format
        gt_geojsons = glob.glob(search)
    len_bootstrap_sample = len(gt_geojsons)
    
    # get objects
    if len(object_subset) == 0:
        if verbose:
            print("getting unique objects...")
        prop_objs, object_subset, all_objs = eval.get_all_objects(
            proposal_polygons_dir, gt_polygons_dir,
            geojson_names_subset=[],
            prediction_cat_attrib=prediction_cat_attrib,
            gt_cat_attrib=gt_cat_attrib, file_format=file_format)
    if verbose:
        print("objects:", object_subset)
    
    # populate out_dict
    
    out_dict = {'mAP': [], 'mF1': []}
    for ob in object_subset:
        out_dict[ob] = {'AP':[], 'F1': []}
        
    for i in range(N_bootstraps):
           
        # sample with replacement
        geojson_names_boot = random.choices(gt_geojsons, k=len_bootstrap_sample)

        if verbose:
            print("\nBootstrap:", i, "/", N_bootstraps)      
            print("  geojson_names_boot[:4]:",  geojson_names_boot[:4]) 
         
        # get performance
        object_classes, mAP, APs_by_class, mF1_score, f1s_by_class, \
            precision_iou_by_obj, precision_by_class, mPrecision, \
            recall_iou_by_obj, recall_by_class, mRecall, confidences = \
                    eval.mAP_score(proposal_polygons_dir, gt_polygons_dir,
                                  geojson_names_subset=geojson_names_boot,
                                  prediction_cat_attrib=prediction_cat_attrib, 
                                  gt_cat_attrib=gt_cat_attrib,
                                  object_subset=object_subset, 
                                  threshold=threshold, 
                                  confidence_attrib=confidence_attrib,
                                  file_format=file_format)
        
        out_dict['mAP'].append(mAP)
        out_dict['mF1'].append(mF1_score)
        # for each object class, add to the dict the performance by class.
        for j, ob in enumerate(object_classes):
            out_dict[ob]['AP'].append(APs_by_class[j])
            out_dict[ob]['F1'].append(f1s_by_class[j])
            
        if ((i % 20) == 0) and verbose:
            print("  ", i, "/", N_bootstraps, "- mAP:", mAP, "mF1:", mF1_score)        
        
    # compute standard deviatios 
    out_dict['mAP_std'] = np.std( out_dict['mAP'])
    out_dict['mF1_std'] = np.std( out_dict['mF1'])
    # get means, stds for each object
    for ob in object_subset:
        ap_mean_name = ob + '_ap_mean'
        ap_std_name = ob + '_ap_std'
        f1_mean_name = ob + '_f1_mean'
        f1_std_name = ob + '_f1_std'
        out_dict[ap_mean_name] = np.mean(out_dict[ob]['AP'])
        out_dict[ap_std_name] = np.std(out_dict[ob]['AP'])
        out_dict[f1_mean_name] = np.mean(out_dict[ob]['F1'])
        out_dict[f1_std_name] = np.std(out_dict[ob]['F1'])
    
    return out_dict
