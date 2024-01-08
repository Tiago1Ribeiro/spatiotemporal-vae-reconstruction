# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation functions.
"""

import numpy as np
from typing import Optional, List, Dict

from shapely.validation import make_valid
from shapely.wkt import loads
from shapely import hausdorff_distance
from shapely import Polygon

from log_setup import logger


def iou_wkt(
    gt_wkt: str,
    model_wkt: str,
    eval_idx: Optional[List[int]] = None,
    discard_100: bool = False,
) -> tuple:
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
            model_wkt = f.read().splitlines()
        # filter indexes to evaluate
        model_wkt = [model_wkt[i] for i in eval_idx]

        if len(ground_truth_wkt) != len(model_wkt):
            logger.warning(
                f"Number of polygons in ground truth ({len(ground_truth_wkt)}) and model ({len(model_wkt)}) do not match."
            )
            if len(ground_truth_wkt) > len(model_wkt):
                ground_truth_wkt = ground_truth_wkt[: len(model_wkt)]
            else:
                model_wkt = model_wkt[: len(ground_truth_wkt)]

    except Exception as e:
        logger.error(f"Error reading WKT files: {e}")
        return

    # Convert WKT lines to list of shapely polygons
    try:
        g_t_polys = [loads(wkt) for wkt in ground_truth_wkt]
        model_polys = [loads(wkt) for wkt in model_wkt]
    except Exception as e:
        logger.error(f"Error converting WKT to polygons: {e}")
        return

    # # Select the first num_polygons polygons
    # g_t_polys = ground_truth_polys[:num_polygons]
    # model_polys = model_polys[:num_polygons]

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
    eval_idx: Optional[List[int]] = None,
    discard_100: bool = False,
) -> tuple:
    """
    Calculates the Hausdorff distance between the ground truth and the model.

    Parameters:
    gt-wkt (str): The path to the ground truth WKT file.
    model_wkt (str): The path to the model WKT file.
    eval_idx (int): The last frame to consider. Defaults to 22500.
    discard_100 (bool): Whether to discard polygons which index number is multiple of 100. Defaults to False.

    Returns:
    tuple: A tuple containing the mean Hausdorff distance and the list of individual Hausdorff distances.
    """

    try:
        # Read WKTs files
        logger.info("Reading WKTs files...")
        with open(gt_wkt, "r") as f:
            ground_truth_wkt = f.read().splitlines()
        with open(model_wkt, "r") as f:
            model_wkt = f.read().splitlines()
        # filter indexes to evaluate
        model_wkt = [model_wkt[i] for i in eval_idx]

        if len(ground_truth_wkt) != len(model_wkt):
            logger.warning(
                f"Number of polygons in ground truth ({len(ground_truth_wkt)}) and model ({len(model_wkt)}) do not match."
            )
            if len(ground_truth_wkt) > len(model_wkt):
                ground_truth_wkt = ground_truth_wkt[: len(model_wkt)]
            else:
                model_wkt = model_wkt[: len(ground_truth_wkt)]

    except Exception as e:
        logger.error(f"Error reading WKT files: {e}")
        return

    # Convert WKT lines to list of shapely polygons
    try:
        g_t_polys = [loads(wkt) for wkt in ground_truth_wkt]
        model_polys = [loads(wkt) for wkt in model_wkt]
    except Exception as e:
        logger.error(f"Error converting WKT to polygons: {e}")
        return e

    # Select the first num_polygons polygons
    # g_t_polys = ground_truth_polys[:last_frame]
    # model_polys = model_polys[:last_frame]

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


# def wkt_hausdorff_distance(ground_truth_wkt: List[str], eval_wkt: List[str]) -> tuple:
#     """
#     Calculate the Hausdorff distance between two sets of polygons.

#     Parameters:
#         ground_truth_wkt (List[str]): List of Well-Known Text (WKT) representations of polygons.
#         eval_wkt (List[str]): List of Well-Known Text (WKT) representations of polygons.

#     Returns:
#         tuple: Tuple containing the mean Hausdorff distance and the list of individual Hausdorff distances.
#     """
#     # Load the polygons from the WKT representations
#     model_polygons = [loads(wkt) for wkt in eval_wkt]
#     ground_truth_polygons = [loads(wkt) for wkt in ground_truth_wkt]

#     # Ensure that the polygons are valid
#     ground_truth_polygons = [make_valid(polygon) for polygon in ground_truth_polygons]
#     model_polygons = [make_valid(polygon) for polygon in model_polygons]

#     # Calculate the Hausdorff distances
#     hausdorff_distances = []
#     for gt_poly, model_poly in zip(ground_truth_polygons, model_polygons):
#         try:
#             hausdorff_dist = gt_poly.hausdorff_distance(model_poly)
#             hausdorff_distances.append(hausdorff_dist)
#         except Exception as e:
#             print(f"Error calculating Hausdorff distance: {e}")

#     # Calculate the mean Hausdorff distance
#     mean_hausdorff_dist = np.mean(hausdorff_distances) if hausdorff_distances else None

#     return mean_hausdorff_dist, hausdorff_distances


def strided_temporal_consistency(
    wkt_file: str, num_polygons: int, stride: int = 1, exp: bool = True
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

    # Load polygons from WKT file
    with open(wkt_file, "r") as f:
        polygons = [loads(line.strip()) for line in f]

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


# def strided_temporal_consistency(
#     wkt_file: str, num_polygons: int, strides: List[int] = [1], exp: bool = True
# ) -> Dict[str, List]:
#     """
#     Calculates the temporal consistency between polygons in a sequence with a certain stride.

#     Parameters:
#     wkt_file (str): The path to the WKT file containing the polygons.
#     num_polygons (int): The total number of polygons in the sequence.
#     strides (List[int], optional): The strides to use. Defaults to [1].
#     exp (bool, optional): Whether to exponentiate the temporal consistency values. Defaults to True.

#     Returns:
#     dict: A dictionary containing the temporal consistency values and related information. The keys are:
#         - i (list): A list of the starting indices of the compared polygon pairs.
#         - strd (int): The stride value used.
#         - tc (list): A list of the temporal consistency values calculated for each pair of polygons.
#         - tc_mean (float): The mean temporal consistency value.

#     """
#     # Load polygons from WKT file
#     with open(wkt_file, "r") as f:
#         polygons = [loads(line.strip()) for line in f]

#     t_c = {key: [] for key in ["i", "strd", "tc", "tc_mean"]}
#     polygons = [make_valid(poly) for poly in polygons]

#     for stride in strides:
#         for i in range(0, num_polygons - stride):
#             t_c["i"].append(i)
#             area_i = polygons[i].area
#             area_next = polygons[i + stride].area
#             diff_area = area_i - area_next
#             tc_temp = 1 - (diff_area / area_next)
#             t_c["tc"].append(np.power(tc_temp, 10) if exp else tc_temp)

#         t_c["tc_mean"].append(np.mean(t_c["tc"][-num_polygons:]))
#         t_c["strd"].append(stride)
#     logger.info(f"Calculated temporal consistency with strides {strides}.")

#     return t_c
