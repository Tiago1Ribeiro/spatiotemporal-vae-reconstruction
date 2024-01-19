# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation functions.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import csv

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


def strided_temp_consistency(
    wkt_file: str,
    strides: List[int],
    interval: List[int],
) -> Dict[str, Dict[str, Any]]:
    """
        Calculate the temporal consistency of a set of polygons within a specified interval.

        This function reads a set of polygons from a WKT file, validates them, and calculates
        their temporal consistency. The calculation is performed with different strides within
        a specified interval of polygons.

    Parameters:
        wkt_file (str): Path to the WKT file containing the polygons.
        strides (List[int]): List of strides to be used in the temporal consistency calculation.
        interval (List[int]): List of two integers specifying the start and end indices of the
        interval within which the temporal consistency should be calculated.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping each stride to another dictionary containing
        the indices of the polygons, the calculated temporal consistencies, the mean temporal
        consistency, and the stride itself.
    """
    logger.info(f"Reading WKT file {wkt_file}...")
    with open(wkt_file, "r") as f:
        polygons = [loads(line.strip()) for line in f]


    # Validate interval
    if interval[0] >= interval[1]:
        logger.error(
            f"Invalid interval: {interval}. End index must be greater than start index."
        )
        return {}

    logger.info(f"Validating polygons...")
    polygons = [make_valid(poly) for poly in polygons[interval[0] : interval[1]]]

    # Validate strides
    for stride in strides:
        if stride > len(polygons):
            logger.warning(
                f"Stride {stride} is greater than the number of polygons ({len(polygons)}). Skipping this stride."
            )
            strides.remove(stride)
    print(f"Valid strides: {strides}")


    results = {}
    for stride in strides:
        logger.info(f"Calculating the temporal consistency with stride {stride}...")
        t_c = {"i": [], "tc": []}
        for i in range(len(polygons) - stride):
            t_c["i"].append(i + interval[0])
            t_c["tc"].append(calculate_temporal_consistency(i, polygons, stride))

        t_c["tc_mean"] = np.mean(t_c["tc"])
        t_c["strd"] = stride
        results[stride] = t_c

    return results


def save_results_to_csv(
    results: Dict[str, Dict[str, Any]],
    base_filename: str,
    fieldnames: Optional[List[str]] = None,
    overwrite: bool = True,
):
    mode = "w" if overwrite else "a"
    with open(base_filename + ".csv", mode, newline="") as csvfile:
        # Use os nomes dos campos fornecidos, se houver
        fieldnames = (
            fieldnames if fieldnames else ["Stride", "Start Index", "Temp. Consistency"]
        )
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Escreva o cabeçalho apenas se o arquivo estiver vazio ou estivermos substituindo o arquivo
        if mode == "w" or csvfile.tell() == 0:
            writer.writeheader()

        for stride, result in results.items():
            for i, tc in zip(result["i"], result["tc"]):
                writer.writerow(
                    {
                        "Stride": stride,
                        "Start Index": i,
                        "Temp. Consistency": tc,
                    }
                )
    logger.info(f"Saved results to {base_filename}.csv.")


def calculate_temporal_consistency(i, polygons, stride, exp=False):
    tc_temp = 1 - (
        polygons[i].difference(polygons[i + stride]).area / polygons[i + stride].area
    )
    return np.power(tc_temp, 10) if exp else tc_temp
