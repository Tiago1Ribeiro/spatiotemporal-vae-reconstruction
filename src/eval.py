# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation functions.
"""

import numpy as np
from typing import Optional, List

from shapely.validation import make_valid, 
from shapely.wkt import loads
from shapely import Polygon

from log_setup import logger


def iou_wkt(
    gt_wkt: str,
    model_wkt: str,
    discard_100: bool = False,
    num_polygons: Optional[int] = None,
) -> float:
    """
    Calculates the Intersection over Union (IoU) between segmentation polygons in WKT format.

    Parameters:
    gt_wkt (str): The path to the ground truth WKT file.
    model_wkt (str): The path to the model WKT file.
    discard_100 (bool): Optional. If True, the polygons which index number is multiple of 100 are discarded.
    num_polygons (int): Optional. If provided, only the first num_polygons polygons are considered.

    Returns:
    float: The mean IoU between the polygons.
    """

    try:
        # Read WKTs files
        logger.info("Reading WKTs files...")
        with open(gt_wkt, "r") as f:
            ground_truth_wkt = f.read().splitlines()

        with open(model_wkt, "r") as f:
            pred_wkt = f.read().splitlines()
    except Exception as e:
        logger.error(f"Error reading WKT files: {e}")
        return

    # Convert WKT lines to list of shapely polygons
    try:
        ground_truth_polys = [loads(wkt) for wkt in ground_truth_wkt]
        model_polys = [loads(wkt) for wkt in pred_wkt]
    except Exception as e:
        logger.error(f"Error converting WKT to polygons: {e}")
        return

    # Select the first num_polygons polygons
    g_t_polys = ground_truth_polys[:num_polygons]
    model_polys = model_polys[:num_polygons]

    # Discard polygons which index number is multiple of 100 if discard_100 is True
    if discard_100:
        g_t_polys = [g_t_polys[i] for i in range(len(g_t_polys)) if i % 100 != 0]
        model_polys = [model_polys[i] for i in range(len(model_polys)) if i % 100 != 0]

    iou_list = []
    logger.info("Calculating IoU...")
    for i in range(len(g_t_polys)):
        g_t_polys[i] = make_valid(g_t_polys[i])
        model_polys[i] = make_valid(model_polys[i])
        intersection = g_t_polys[i].intersection(model_polys[i]).area
        union = g_t_polys[i].union(model_polys[i]).area
        iou = intersection / union
        iou_list.append(iou)

    # Calculate the mean IoU
    iou_mean = np.mean(iou_list)

    return iou_mean, iou_list


def hausdorff_dist_wkt(
    gt_wkt: str,
    model_wkt: str,
    last_frame: Optional[int] = 22500,
    discard_100: bool = False,
) -> float:
    """
    Calculates the Hausdorff distance between the ground truth and the model.

    Parameters:
    ground_truth_file (str): The path to the ground truth WKT file.
    model_file (str): The path to the model WKT file.
    last_frame (int): The last frame to consider. Defaults to 22500.
    discard_100 (bool): Whether to discard polygons which index number is multiple of 100. Defaults to False.

    Returns:
    float: The mean Hausdorff distance between the polygons.
    """

    try:
        # Read WKTs files
        logger.info("Reading WKTs files...")
        with open(gt_wkt, "r") as f:
            ground_truth_wkt = f.read().splitlines()

        with open(model_wkt, "r") as f:
            model_wkt = f.read().splitlines()
    except Exception as e:
        logger.error(f"Error reading WKT files: {e}")
        return

    # Convert WKT lines to list of shapely polygons
    try:
        ground_truth_polys = [loads(wkt) for wkt in ground_truth_wkt]
        model_polys = [loads(wkt) for wkt in model_wkt]
    except Exception as e:
        logger.error(f"Error converting WKT to polygons: {e}")
        return e

    # Select the first last_frame polygons
    g_t_polys = ground_truth_polys[:last_frame]
    model_polys = model_polys[:last_frame]

    # Discard polygons which index number is multiple of 100 if discard_100 is True
    if discard_100:
        g_t_polys = [g_t_polys[i] for i in range(len(g_t_polys)) if i % 100 != 0]
        model_polys = [model_polys[i] for i in range(len(model_polys)) if i % 100 != 0]

    hausdorff_dist_list = list()
    logger.info("Calculating Hausdorff Distance...")
    for i in range(len(g_t_polys)):
        g_t_polys[i] = make_valid(g_t_polys[i])
        model_polys[i] = make_valid(model_polys[i])
        hausdorff_dist = g_t_polys[i].hausdorff_distance(model_polys[i])
        hausdorff_dist_list.append(hausdorff_dist)

    # Calculate the mean Hausdorff distance
    hausdorff_dist_mean = sum(hausdorff_dist_list) / len(hausdorff_dist_list)

    return hausdorff_dist_mean, hausdorff_dist_list



def strided_temporal_consistency(
    polygons: List[Polygon], num_polygons: int, stride: int = 1, exp: bool = True
) -> dict:
    """
    Calculates the temporal consistency between polygons in a sequence with a certain stride.

    Parameters:
    polygons (list): A list of Shapely Polygon objects representing the sequence of polygons.
    num_polygons (int): The total number of polygons in the sequence.
    stride (int, optional): The number of polygons to skip between comparisons. Defaults to 1.
    exp (bool, optional): Whether to exponentiate the temporal consistency values. Defaults to True.

    Returns:
    dict: A dictionary containing the temporal consistency values and related information. The keys are:
        - i (list): A list of the starting indices of the compared polygon pairs.
        - strd (int): The stride value used.
        - tc (list): A list of the temporal consistency values calculated for each pair of polygons.
        - tc_mean (float): The mean temporal consistency value.
    """

    # Create empty dictionary to store the TC values and and indexes
    t_c = {key: [] for key in ["i", "strd", "tc", "tc_mean"]}

    # Make all polygons in the list valid before calculating temporal consistency
    polygons = [make_valid(poly) for poly in polygons]

    logger.info(f"Calculating the temporal consistency with stride {stride}...")
    for i in range(0, num_polygons - stride):
        # Calculates the temporal consistency between two consecutive polygons
        t_c["i"].append(i)
        tc_temp = 1 - (
            polygons[i].difference(polygons[i + stride]).area
            / polygons[i + stride].area
        )
        t_c["tc"].append(np.power(tc_temp, 10) if exp else tc_temp)

    # Calculates the mean of the temporal consistency
    t_c["tc_mean"] = np.mean(t_c["tc"])
    t_c["strd"] = stride

    return t_c
