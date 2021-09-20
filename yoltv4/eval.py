'''
Eval functions stem from the excellent Solaris package. We opt to copy the 
evaluation functions here rather than include Solaris as part of YOLTv4 in 
order to reduce install complexity and reduce the docker container size. 
Functions are lightly modified for increased output fidelity.
See https://github.com/CosmiQ/solaris for documentation. 
'''

import os
import glob
from tqdm import tqdm
import numpy as np
import geopandas as gpd
# needed for obj_confusion_matrix and plotting
import pandas as pd
import matplotlib.pyplot as plt


# https://github.com/CosmiQ/solaris/blob/dev/solaris/eval/iou.py
def calculate_iou(pred_poly, test_data_GDF):
    """Get the best intersection over union for a predicted polygon.
    Arguments
    ---------
    pred_poly : :py:class:`shapely.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.
    Returns
    -------
    iou_GDF : :py:class:`geopandas.GeoDataFrame`
        A subset of ``test_data_GDF`` that overlaps ``pred_poly`` with an added
        column ``iou_score`` which indicates the intersection over union value.
    """

    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = test_data_GDF[test_data_GDF.intersects(pred_poly)]

    iou_row_list = []
    for _, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
        else:
            iou_score = 0
        row['iou_score'] = iou_score
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF


# https://github.com/CosmiQ/solaris/blob/dev/solaris/eval/vector.py
def average_score_by_class(ious, threshold=0.5):
    """ for a list of object ious by class, test if they are a counted as a
    positive or a negative.
    Arguments
    ---------
        ious : list of lists
            A list containing individual lists of ious for eachobject class.
        threshold : float
            A value between 0.0 and 1.0 that determines the threshold for a true positve.
    Returns
    ---------
        average_by_class : list
            A list containing the ratio of true positives for each class
    """
    binary_scoring_lists = []
    for x in ious:
        items = []
        for i in x:
            if i >= threshold:
                items.append(1)
            else:
                items.append(0)
        binary_scoring_lists.append(items)
    average_by_class = []
    for l in binary_scoring_lists:
        average_by_class.append(np.nanmean(l))
    return average_by_class


# https://github.com/CosmiQ/solaris/blob/dev/solaris/eval/vector.py
def get_all_objects(proposal_polygons_dir, gt_polygons_dir,
                    geojson_names_subset=[],
                    prediction_cat_attrib="class", gt_cat_attrib='make',
                    file_format="geojson", verbose=False):
    """ Using the proposal and ground truth polygons, calculate the total.
    Filenames of predictions and ground-truth must be identical.
    unique classes present in each
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
        file_format : str
            The extension or file format for predictions
    Returns
    ---------
            prop_objs : list
                All unique objects that exist in the proposals
            gt_obj : list
                All unique objects that exist in the ground truth
            all_objs : list
                A union of the prop_objs and gt_objs lists
    """
                    
    # proposal objects
    if len(geojson_names_subset) > 0:
        os.chdir(proposal_polygons_dir)
        proposal_geojsons = geojson_names_subset
    else:
        os.chdir(proposal_polygons_dir)
        search = "*" + file_format
        proposal_geojsons = glob.glob(search)
    objs = []        
    for geojson in tqdm(proposal_geojsons):
        # print("geojson:", geojson)
        ground_truth_poly = os.path.join(gt_polygons_dir, geojson)
        if os.path.exists(ground_truth_poly):
            ground_truth_gdf = gpd.read_file(ground_truth_poly)
            proposal_gdf = gpd.read_file(geojson)
            for index, row in (proposal_gdf.iterrows()):
                objs.append(row[prediction_cat_attrib])
    prop_objs = sorted(list(set(objs)))
    if verbose:
        print("prop_objs:", prop_objs)

    # gt objects
    if len(geojson_names_subset) > 0:
        gt_geojsons = geojson_names_subset
        os.chdir(gt_polygons_dir)
    else:
        os.chdir(gt_polygons_dir)
        search = "*" + file_format
        gt_geojsons = glob.glob(search)
    objs = []
    for geojson in tqdm(gt_geojsons):
        proposal_poly = os.path.join(proposal_polygons_dir, geojson)
        if os.path.exists(proposal_poly):
            proposal_gdf = gpd.read_file(proposal_poly)
            ground_truth_gdf = gpd.read_file(geojson)
            for index, row in (ground_truth_gdf.iterrows()):
                objs.append(row[gt_cat_attrib])
    gt_objs = sorted(list(set(objs)))
    if verbose:
        print("gt_objs:", gt_objs)
    

    # all objs
    all_objs = gt_objs + prop_objs
    all_objs = sorted(list(set(all_objs)))
    if verbose:
        print("all_objs:", all_objs)
    
    return prop_objs, gt_objs, all_objs


# https://github.com/CosmiQ/solaris/blob/dev/solaris/eval/vector.py
def precision_calc(proposal_polygons_dir, gt_polygons_dir,
                   geojson_names_subset=[],
                   prediction_cat_attrib="class", gt_cat_attrib='make', confidence_attrib=None,
                   object_subset=[], threshold=0.5, file_format="geojson"):
    """ Using the proposal and ground truth polygons, calculate precision metrics.
    Filenames of predictions and ground-truth must be identical.  Will only
    calculate metric for classes that exist in the ground truth.
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
        confidence_attrib : str
            The column or attribute within the proposal polygons that
            specifies model confidence for each prediction
        object_subset : list
            A list or subset of the unique objects that are contained within the
            ground truth polygons. If empty, this will be
            auto-created using all classes that appear ground truth polygons.
        threshold : float
            A value between 0.0 and 1.0 that determines the IOU threshold for a
            true positve.
        file_format : str
            The extension or file format for predictions
    Returns
    ---------
        object_subset: list of categories
        iou_holder : list of lists
            An iou score for each object per class (precision specific)
        precision_by_class : list
            A list containing the precision score for each class
        mPrecision : float
            The mean precision score of precision_by_class
        confidences : list of lists
            All confidences for each object for each class
    """

    if len(geojson_names_subset) > 0:
        os.chdir(proposal_polygons_dir)
        proposal_geojsons = geojson_names_subset
        # make sure proposal geojsons exist
        proposal_geojsons = []
        for name_tmp in geojson_names_subset:
            if os.path.exists(os.path.join(proposal_polygons_dir, name_tmp)):
                proposal_geojsons.append(name_tmp)
            else:
                print("Warning, missing proposal file for precision_calc():", name_tmp)
    else:
        os.chdir(proposal_polygons_dir)
        search = "*" + file_format
        proposal_geojsons = glob.glob(search)

    ious = []
    iou_holder = []
    confidences = []
    if len(object_subset) == 0:
        prop_objs, object_subset, all_objs = get_all_objects(
            proposal_polygons_dir, gt_polygons_dir,
            geojson_names_subset=geojson_names_subset,
            prediction_cat_attrib=prediction_cat_attrib,
            gt_cat_attrib=gt_cat_attrib, file_format=file_format)
    for i in range(len(object_subset)):
        iou_holder.append([])
        confidences.append([])

    for geojson in tqdm(proposal_geojsons):
        ground_truth_poly = os.path.join(gt_polygons_dir, geojson)
        if os.path.exists(ground_truth_poly):
            ground_truth_gdf = gpd.read_file(ground_truth_poly)
            proposal_gdf = gpd.read_file(geojson)
            #tmp print(" proposal_gdf.head():", proposal_gdf.head())
            i = 0
            for obj in object_subset:
                # print("  obj:", obj)
                conf_holder = []
                proposal_gdf2 = proposal_gdf[proposal_gdf[prediction_cat_attrib] == obj]
                #tmp print("   len proposal_gdf2:", len(proposal_gdf2))
                for index, row in (proposal_gdf2.iterrows()):
                    if confidence_attrib is not None:
                        conf_holder.append(row[confidence_attrib])
                    iou_GDF = calculate_iou(row.geometry, ground_truth_gdf)
                    #tmp print("iou_GDF.columns:", iou_GDF.columns)
                    if 'iou_score' in iou_GDF.columns:
                        iou = iou_GDF.iou_score.max()
                        max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]
                        id_1 = row[prediction_cat_attrib]
                        id_2 = ground_truth_gdf.loc[max_iou_row.name][gt_cat_attrib]
                        if id_1 == id_2:
                            ious.append(iou)
                            ground_truth_gdf.drop(max_iou_row.name, axis=0, inplace=True)
                        else:
                            iou = 0
                            ious.append(iou)
                    else:
                        iou = 0
                        ious.append(iou)
                for item in ious:
                    iou_holder[i].append(item)
                if confidence_attrib is not None:
                    for conf in conf_holder:
                        confidences[i].append(conf)
                ious = []
                i += 1
        else:
            print("Warning- No ground truth for:", geojson)
            proposal_gdf = gpd.read_file(geojson)
            i = 0

            for obj in object_subset:
                proposal_gdf2 = proposal_gdf[proposal_gdf[gt_cat_attrib] == obj]
                for z in range(len(proposal_gdf2)):
                    ious.append(0)
                for item in ious:
                    iou_holder[i].append(item)
                if confidence_attrib is not None:
                    for conf in conf_holder:
                        confidences[i].append(conf)
                i += 1
                ious = []
    precision_by_class = average_score_by_class(iou_holder, threshold=threshold)
    precision_by_class = list(np.nan_to_num(precision_by_class))
    mPrecision = np.nanmean(precision_by_class)
    print("mPrecision:", mPrecision)

    return object_subset, iou_holder, precision_by_class, mPrecision, confidences


# https://github.com/CosmiQ/solaris/blob/dev/solaris/eval/vector.py
def recall_calc(proposal_polygons_dir, gt_polygons_dir,
                geojson_names_subset=[],
                prediction_cat_attrib="class", gt_cat_attrib='make',
                object_subset=[], threshold=0.5, file_format="geojson"):
    """ Using the proposal and ground truth polygons, calculate recall metrics.
    Filenames of predictions and ground-truth must be identical. Will only
    calculate metric for classes that exist in the ground truth.
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
            ground truth polygons. If empty, this will be
            auto-created using all classes that appear ground truth polygons.
        threshold : float
            A value between 0.0 and 1.0 that determines the IOU threshold for a
            true positve.
        file_format : str
            The extension or file format for predictions
    Returns
    ---------
        object_subset: list of categories
        iou_holder : list of lists
            An iou score for each object per class (recall specific)
        recall_by_class : list
            A list containing the recall score for each class
        mRecall : float
            The mean recall score of recall_by_class
    """

    if len(geojson_names_subset) > 0:
        os.chdir(gt_polygons_dir)
        gt_geojsons = geojson_names_subset
    else:
        os.chdir(gt_polygons_dir)
        search = "*" + file_format
        gt_geojsons = glob.glob(search)

    ious = []
    iou_holder = []
    if len(object_subset) == 0:
        prop_objs, object_subset, all_objs = get_all_objects(
            proposal_polygons_dir, gt_polygons_dir,
            geojson_names_subset=geojson_names_subset,
            prediction_cat_attrib=prediction_cat_attrib,
            gt_cat_attrib=gt_cat_attrib, file_format=file_format)
    for i in range(len(object_subset)):
        iou_holder.append([])
    for geojson in tqdm(gt_geojsons):
        proposal_poly = os.path.join(proposal_polygons_dir, geojson)
        if os.path.exists(proposal_poly):
            proposal_gdf = gpd.read_file(proposal_poly)
            ground_truth_gdf = gpd.read_file(geojson)
            i = 0
            for obj in object_subset:
                ground_truth_gdf2 = ground_truth_gdf[ground_truth_gdf[gt_cat_attrib] == obj]
                for index, row in (ground_truth_gdf2.iterrows()):
                    iou_GDF = calculate_iou(row.geometry, proposal_gdf)
                    if 'iou_score' in iou_GDF.columns:
                        iou = iou_GDF.iou_score.max()
                        max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]
                        id_1 = row[gt_cat_attrib]
                        id_2 = proposal_gdf.loc[max_iou_row.name][prediction_cat_attrib]
                        if id_1 == id_2:
                            ious.append(iou)
                            proposal_gdf.drop(max_iou_row.name, axis=0, inplace=True)
                        else:
                            iou = 0
                            ious.append(iou)
                    else:
                        iou = 0
                        ious.append(iou)
                for item in ious:
                    iou_holder[i].append(item)
                i += 1
                ious = []
        else:
            ground_truth_gdf = gpd.read_file(geojson)
            i = 0
            for obj in object_subset:
                ground_truth_gdf2 = ground_truth_gdf[ground_truth_gdf[gt_cat_attrib] == obj]
                for z in range(len(ground_truth_gdf2)):
                    ious.append(0)
                for item in ious:
                    iou_holder[i].append(item)
                i += 1
                ious = []

    recall_by_class = average_score_by_class(iou_holder, threshold=threshold)
    recall_by_class = list(np.nan_to_num(recall_by_class))
    mRecall = np.nanmean(recall_by_class)
    print("mRecall:", mRecall)
    return object_subset, iou_holder, recall_by_class, mRecall


# https://github.com/CosmiQ/solaris/blob/dev/solaris/eval/vector.py
def mF1(proposal_polygons_dir, gt_polygons_dir, 
        geojson_names_subset=[],
        prediction_cat_attrib="class",
        gt_cat_attrib='make', object_subset=[], threshold=0.5, 
        confidence_attrib='prob',
        file_format="geojson", all_outputs=False):
    """ Using the proposal and ground truth polygons, calculate F1 and mF1
    metrics. Filenames of predictions and ground-truth must be identical.  Will
    only calculate metric for classes that exist in the ground truth.
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
        all_outputs : bool
            `True` or `False`.  If `True` returns an expanded output.
    Returns
    ---------
        if all_outputs is `True`:
            object_subset : list
                List of objects
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
        if all_outputs is `False`:
            mF1_score : float
                The mean F1 score of f1s_by_class (only calculated for ground
                ground truth classes)
            f1s_by_class : list
                A list containing the f1 score for each class
    """
    if len(object_subset) == 0:
        print("getting unique objects...")
        prop_objs, object_subset, all_objs = get_all_objects(
            proposal_polygons_dir, gt_polygons_dir,
            geojson_names_subset=geojson_names_subset,
            prediction_cat_attrib=prediction_cat_attrib,
            gt_cat_attrib=gt_cat_attrib, file_format=file_format)
    print("calculating recall...")
    _, recall_iou_by_obj, recall_by_class, mRecall = recall_calc(
        proposal_polygons_dir, gt_polygons_dir,
        geojson_names_subset=geojson_names_subset,
        prediction_cat_attrib=prediction_cat_attrib,
        gt_cat_attrib=gt_cat_attrib, 
        object_subset=object_subset,
        threshold=threshold, file_format=file_format)
    print("calculating precision...")
    _, precision_iou_by_obj, precision_by_class, mPrecision, confidences = precision_calc(
        proposal_polygons_dir, gt_polygons_dir,
        geojson_names_subset=geojson_names_subset,
        prediction_cat_attrib=prediction_cat_attrib,
        gt_cat_attrib=gt_cat_attrib, 
        confidence_attrib=confidence_attrib,
        object_subset=object_subset,
        threshold=threshold, 
        file_format=file_format)
                
    print("calculating F1 scores...")
    f1s_by_class = []
    for recall, precision in zip(recall_by_class, precision_by_class):
        f1 = 2 * precision * recall / (precision + recall)
        f1 = np.nan_to_num(f1)
        f1s_by_class.append(f1)
    mF1_score = np.nanmean(f1s_by_class)
    print("mF1:", mF1_score)
    if all_outputs is True:
        return object_subset, mF1_score, f1s_by_class, precision_iou_by_obj, precision_by_class, mPrecision, recall_iou_by_obj, recall_by_class, mRecall, confidences
    else:
        return mF1_score, f1s_by_class


# https://github.com/CosmiQ/solaris/blob/dev/solaris/eval/vector.py
def mAP_score(proposal_polygons_dir, gt_polygons_dir,
              geojson_names_subset=[],
              prediction_cat_attrib="class", gt_cat_attrib='make',
              object_subset=[], threshold=0.5, confidence_attrib='prob',
              file_format="geojson", verbose=False):
    """ Using the proposal and ground truth polygons calculate the Mean Average
    Precision (mAP) and  mF1 metrics. Filenames of predictions and ground-truth
    must be identical.  Will only calculate metric for classes that exist in
    the ground truth.

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
    
    object_subset, mF1_score, f1s_by_class, precision_iou_by_obj, precision_by_class, mPrecision, recall_iou_by_obj, recall_by_class, mRecall, confidences = mF1(
        proposal_polygons_dir, gt_polygons_dir,
        geojson_names_subset=geojson_names_subset,
        prediction_cat_attrib=prediction_cat_attrib,
        gt_cat_attrib=gt_cat_attrib, object_subset=object_subset,
        threshold=threshold, confidence_attrib=confidence_attrib,
        file_format=file_format, all_outputs=True)
    if verbose:
        print("object_subset:", object_subset)
    recall_thresholds = np.arange(0, 1.01, 0.01).tolist()
    APs_by_class = []
    for p_obj_list, c_obj_list, r_obj_list in zip(precision_iou_by_obj, confidences, recall_iou_by_obj):
        num_objs = len(r_obj_list)
        p_obj_list_sorted = [x for _, x in sorted(zip(c_obj_list, p_obj_list))]
        p_obj_list_sorted.reverse()
        TPs = []
        FPs = []
        for p in p_obj_list_sorted:
            if p >= threshold:
                TPs.append(1)
                FPs.append(0)
            else:
                TPs.append(0)
                FPs.append(1)
        Acc_TPs = []
        Acc_FPs = []
        t_sum = 0
        f_sum = 0
        for t, f in zip(TPs, FPs):
            t_sum += t
            f_sum += f
            Acc_TPs.append(t_sum)
            Acc_FPs.append(f_sum)
        precisions = []
        recalls = []

        for aTP, aFP in zip(Acc_TPs, Acc_FPs):
            precision = (aTP / (aTP + aFP))
            precisions.append(precision)
            if num_objs > 0:
                recall = (aTP / num_objs)
            else:
                recall = 0
            recalls.append(recall)
        interp = []
        for t in recall_thresholds:
            precisions2 = [p for r, p in zip(recalls, precisions) if r >= t]
            if len(precisions2) > 0:
                interp.append(np.max(precisions2))
            else:
                interp.append(0)

        AP = np.average(interp)
        APs_by_class.append(AP)
    mAP = np.average(APs_by_class)
    print("mAP:", mAP, "@IOU:", threshold)
    return object_subset, mAP, APs_by_class, mF1_score, f1s_by_class, precision_iou_by_obj, precision_by_class, mPrecision, recall_iou_by_obj, recall_by_class, mRecall, confidences
        

def obj_confusion_matrix(proposal_polygons_dir, gt_polygons_dir,
                   geojson_names_subset=[],
                   prediction_cat_attrib="class", gt_cat_attrib='make', confidence_attrib=None,
                   object_subset=[], threshold=0.5, file_format="geojson", verbose=False, 
                   super_verbose=False):
    """ Using the proposal and ground truth polygons, calculate confusion matrix.
    Adapted from https://github.com/CosmiQ/solaris/blob/dev/solaris/eval/vector.py precision_calc()
    Filenames of predictions and ground-truth must be identical.  Will only
    calculate metric for classes that exist in the ground truth.
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
        confidence_attrib : str
            The column or attribute within the proposal polygons that
            specifies model confidence for each prediction
        object_subset : list
            A list or subset of the unique objects that are contained within the
            ground truth polygons. If empty, this will be
            auto-created using all classes that appear ground truth polygons.
        threshold : float
            A value between 0.0 and 1.0 that determines the IOU threshold for a
            true positve.
        file_format : str
            The extension or file format for predictions
        verbose : bool
            Switch to print all the progress
    Returns
    ---------
        out_gdf: geodataframe
            geodataframe containing matches between ground truth and proposal polygons
            columns = ['prop_cat', 'prop_prob', 'prop_geom', 'gt_cat', 'gt_geom', 'iou', 'geojson']
        summary_df: dataframe
            dataframe containing aggregated stats for each image
            columns = ['geojson', 'n_gt', 'n_prop', 'n_fn', 'n_fp', 'n_matched']

    """

    from shapely.geometry import Point
    out_columns = ['prop_cat', 'prop_prob', 'prop_geom', 'gt_cat', 'gt_geom', 'iou', 'geojson']
    summary_columns = ['geojson', 'n_gt', 'n_prop', 'n_fn', 'n_fp', 'n_matched']

    if len(geojson_names_subset) > 0:
        os.chdir(proposal_polygons_dir)
        proposal_geojsons = geojson_names_subset
        # make sure proposal geojsons exist
        proposal_geojsons = []
        for name_tmp in geojson_names_subset:
            if os.path.exists(os.path.join(proposal_polygons_dir, name_tmp)):
                proposal_geojsons.append(name_tmp)
            else:
                print("Warning, missing proposal file for precision_calc():", name_tmp)
    else:
        os.chdir(proposal_polygons_dir)
        search = "*" + file_format
        proposal_geojsons = glob.glob(search)

    # get objects, if not passed as argument
    if len(object_subset) == 0:
        prop_objs, object_subset, all_objs = get_all_objects(
            proposal_polygons_dir, gt_polygons_dir,
            geojson_names_subset=geojson_names_subset,
            prediction_cat_attrib=prediction_cat_attrib,
            gt_cat_attrib=gt_cat_attrib, file_format=file_format)
    print("object_subset for obj_confusion_matrix():", object_subset)

    out_arr = []
    summary_arr = []
    for geojson in tqdm(proposal_geojsons):
        ground_truth_poly = os.path.join(gt_polygons_dir, geojson)
        proposal_poly = os.path.join(proposal_polygons_dir, geojson)
        if super_verbose:
            print("\n", geojson)
        if os.path.exists(ground_truth_poly):
            ground_truth_gdf = gpd.read_file(ground_truth_poly)
            proposal_gdf = gpd.read_file(proposal_poly)
            # if verbose:
            #     print("prop geojson path:", proposal_poly)
            #    print(" ground_truth_gdf.head():", ground_truth_gdf.head())
            #    print(" proposal_gdf.head():", proposal_gdf.head())
            i = 0
            matched_gt_idxs = []
            n_fp = 0
            for j, obj in enumerate(object_subset):
                if super_verbose:
                    print(" ", j, "/", len(object_subset), "obj:", obj)
                conf_holder = []
                proposal_gdf2 = proposal_gdf[proposal_gdf[prediction_cat_attrib] == obj]
                ground_truth_gdf2 = ground_truth_gdf[ground_truth_gdf[gt_cat_attrib] == obj]            
                #tmp print("   len proposal_gdf2:", len(proposal_gdf2))
                # get all proposals
                for index, row in (proposal_gdf2.iterrows()):
                    prop_geom = row['geometry']
                    prop_cat = row[prediction_cat_attrib]
                    if confidence_attrib is not None:
                        prop_prob = row[confidence_attrib]
                    else: 
                        prop_prob = 0
                    # iou_GDF is the ground_grouth_gdf that intersects the proposal
                    iou_GDF = calculate_iou(row.geometry, ground_truth_gdf)

                    if 'iou_score' in iou_GDF.columns:
                        iou = iou_GDF.iou_score.max()
                        if iou >= threshold:
                            max_iou_idx = iou_GDF['iou_score'].idxmax(axis=0, skipna=True)
                            max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]
                            gt_cat = ground_truth_gdf.loc[max_iou_row.name][gt_cat_attrib]
                            gt_geom = max_iou_row['geometry']
                            matched_gt_idxs.append(max_iou_idx)
                            # if verbose:
                            #    # print("iou_gdf:", iou_GDF)
                            #    print("max_iou_idx:", max_iou_idx)
                            #    pass
                        else:
                            # if iou under threshold, count as false positive
                            gt_cat = 'FP'
                            gt_geom = Point(0,0)  
                            n_fp += 1                  
                    else:
                        # create equival/ent of max_iou_row but record a false positive
                        iou = 0.0
                        gt_cat = 'FP'
                        gt_geom = Point(0,0)
                        n_fp += 1

                    out_row = [prop_cat, prop_prob, prop_geom, gt_cat, gt_geom, iou, geojson]
                    out_arr.append(out_row)
                
            # compare number matched proposals for this object, count false negatives
            # get unique matches
            matched_gt_idxs = np.unique(matched_gt_idxs)
            n_false_neg = max(0, len(ground_truth_gdf) - len(matched_gt_idxs))
            # add false negatives to array
            for idx_tmp in matched_gt_idxs:
                gt_cat = ground_truth_gdf.loc[idx_tmp, gt_cat_attrib]
                gt_geom = ground_truth_gdf.loc[idx_tmp, 'geometry']
                # gt_geom = Point(0,0)  # placeholder geom (decided not to put in the effort to track gt_geom)
                prop_cat, prop_prob, iou, prop_geom = 'FN', 0.0, 0.0, Point(0,0)
                out_arr.append([prop_cat, prop_prob, prop_geom, gt_cat, gt_geom, iou, geojson])
            
            # add to summary_arr (['geojson', 'n_gt', 'n_prop', 'n_fn', 'n_fp', 'n_matched'])
            summary_arr.append([geojson, len(ground_truth_gdf), len(proposal_gdf), 
                                n_false_neg, n_fp, len(matched_gt_idxs)])
            # print, if desired
            if verbose:
                print(geojson, " len(ground_truth_gdf):", len(ground_truth_gdf),
                     " len(prop gdf):", len(proposal_gdf), " n_false_neg:", n_false_neg, 
                     " n_false_pos:", n_fp, " matched_count:", len(matched_gt_idxs))            
            i += 1
        else:
            print("Warning - No ground truth for:", geojson)
            return
     
    # create gdf
    out_gdf = gpd.GeoDataFrame(out_arr, columns=out_columns)
    summary_df = pd.DataFrame(summary_arr, columns=summary_columns)

    return out_gdf, summary_df


def plot_obj_cm(out_gdf, prop_cat_col='prop_cat', gt_cat_col='gt_cat', labels=[], 
    colorbar=False, use_seaborn=False, figsize=(12, 12), bolden_axes_labels=False, show_plot=True):
    """
    Take output of obj_confusion_matrix, and plot the results.
    If no labels provided, arange alphabetically.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    """

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cmap=plt.cm.Reds

    y_gt =   out_gdf[gt_cat_col].values
    y_prop = out_gdf[prop_cat_col].values
    # np.set_printoptions(precision=2)

    if len(labels) == 0:
        # get unique labels
        all_labels = np.append(y_gt, y_prop)
        labels = sorted(np.unique(all_labels))
        # # put certain items at end?
        # if 'Other' in labels:
        #     append_labels = ['Other', 'FN', 'FP']
        #     for l_tmp in append_labels:
        #         if l_tmp in labels:
        #             labels.remove(l_tmp)
        #             labels.extend([l_tmp])
        #             labels.extend([l_tmp])
    else:
        pass
    print("labels:", labels)
    
    norms = [None, 'pred', 'true']
    title_strings = ['Prediction Confusion Matrix', 'Prediction Confusion Matrix (Normalized by Predictions)',
                    'Prediction Confusion Matrix (Normalized by Ground Truth)']

    for norm, title in zip(norms, title_strings):

        fig, ax = plt.subplots(figsize=figsize)
        cm = confusion_matrix(y_gt, y_prop, labels=labels, normalize=norm)

        if not use_seaborn:
            # matplotlib display
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            _ = disp.plot(ax=ax, colorbar=colorbar, cmap=cmap)
            xlocs, xlabels = plt.xticks()
            ax.set_xticklabels(xlabels, rotation=80, color='black')

        else:
            import seaborn
            # seaborn - show all 
            sns.heatmap(cm, annot=True, cmap=plt.cm.Reds)
            # # seaborn - show only half
            # mask = np.zeros_like(cm)
            # mask[np.triu_indices_from(mask, k=1)] = True
            # with sns.axes_style("white"):
            #     ax = sns.heatmap(cm, mask=mask, square=True, annot=True, cmap=plt.cm.Reds)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            # plt.xticks(len(labels), labels, rotation=80)
            
        # make axes bold ?
        if bolden_axes_labels:
            ax_labels = ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.get_label()] + [ax.yaxis.get_label()]
            [label.set_fontweight('bold') for label in ax_labels]
    
        # make axes labels a little bigger?
        ax_labels = [ax.xaxis.get_label()] + [ax.yaxis.get_label()]
        for label in ax_labels:
            orig_size = label.get_fontsize()
            new_size = int(1.3 + orig_size)
            label.set_fontsize(new_size)
            
        # [label.set_fontweight('bold') for label in ax_labels]
       #  orig_size = ax.xaxis.get_label().get_fontsize()
       #  new_size = int(1.2 + orig_size)
       #  ax.set_xlabel('x-axis', fontsize=new_size)
       #  ax.set_ylabel('y-axis', fontsize=new_size)
       #           
        # plt.tight_layout()
        plt.title(title)
        # plt.title("Prediction Confusion Matrix" + "  (norm = " + str(norm) + ")")
        if show_plot:
            plt.show()

    return fig, ax
