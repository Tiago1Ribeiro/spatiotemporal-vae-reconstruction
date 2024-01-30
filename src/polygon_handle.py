#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
from rasterio.features import shapes
from rasterio import Affine
from shapely.geometry import shape, MultiPolygon
from log_setup import logger
from scipy.spatial.distance import jaccard
import numpy as np
from tqdm import tqdm
from rasterio.features import rasterize


def mask_to_poly(mask_img: np.ndarray) -> MultiPolygon:
    """
    Converts a segmentation mask to a shapely multipolygon.

    Parameters:
    mask_img (numpy.ndarray): The segmentation mask.

    Returns:
    shapely.geometry.MultiPolygon: The shapely multipolygon.
    """
    all_polygons = list()

    for shp, _ in shapes(
        source=mask_img.astype(np.uint8),
        mask=(mask_img > 0),
        transform=Affine(1.0, 0, 0, 0, 1.0, 0),
    ):
        all_polygons.append(shape(shp))

    all_polygons = MultiPolygon(all_polygons)

    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        if all_polygons.geom_type == "Polygon":
            all_polygons = MultiPolygon([all_polygons])

    return all_polygons


def save_polygon_to_wkt(polygon: MultiPolygon, file_path: str) -> None:
    """
    Save a shapely polygon to a WKT format file.

    Parameters:
    polygon (shapely.geometry.MultiPolygon): The shapely multipolygon.
    file_path (str): Path to the output file.
    """
    with open(file_path, "a") as f:  # Open in append mode
        f.write(polygon.wkt + "\n")


def masks_to_polygons(
    msks_paths: list,
    out_dim: tuple = (512, 512),
    save_path: str = None,
    return_pol_list: bool = False,
) -> list:
    """
    Converts a list of paths to segmentation masks into a list of shapely multipolygons.

    Parameters:
    msks_paths (list): A list of paths to the masks (list of path to PNG, JPG or TIFF files).
    out_dim (tuple): The desired dimensions of the masks. Default is (512, 512).
    save_path (str): Optional. If provided, the function saves the polygons to a WKT file.
    return_pol_list (bool): Optional. If True, the function returns a list of shapely
        multipolygons. Default is False.

    Returns:
    list: A list of shapely multipolygons.
    """
    start_time = time.time()

    logger.info("Converting masks to polygons...")

    i = 0
    for img_path in msks_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        if (w, h) != out_dim:
            img = cv2.resize(img, out_dim, interpolation=cv2.INTER_CUBIC)
        polygon = mask_to_poly(img)

        if save_path:
            save_polygon_to_wkt(polygon, save_path)

        i += 1
        elapsed_time = time.time() - start_time
        print(
            f"Processed {i} masks out of {len(msks_paths)} | "
            f"Time elapsed: {elapsed_time:.2f}s  ",
            end="\r",
        )
    logger.info(f"Saved polygons to {save_path}")

    if return_pol_list:
        # read the polygons from the file
        pol_list = list()
        with open(save_path, "r") as f:
            for line in f:
                pol_list.append(shape(line))
        return pol_list


def calculate_distances(polygons: list, out_shape: tuple = (720, 1280)):
    """
    Calculates the Jaccard distance between binary segmentation masks of the
    first and subsequent polygons

    Args:
        polygons (list): A list of polygons represented as a list of coordinate
        tuples.
        out_shape (tuple): The shape of the output rasterized mask.
        Default is (720, 1280).

    Returns:
        dict: A dictionary with two keys - "Jaccard distance". The values
        for each key are lists containing the distance values between the
        first and subsequent polygons

    """
    distances = {"Jaccard distance": []}
    masks = rasterize(polygons, out_shape)
    mask_t0 = masks[0]

    # Calculate Jaccard distances
    for i in tqdm(range(1, len(polygons))):
        mask_tn = masks[i]
        if np.all(mask_tn == 0) or np.all(mask_t0 == 0):
            distances["Jaccard distance"].append(0)
        else:
            jaccard_distance = jaccard(mask_t0.flatten(), mask_tn.flatten())
            distances["Jaccard distance"].append(jaccard_distance)

    return distances


def gen_similar_poly_samples(polygons, threshold=0.15, out_shape=(720, 1280)):
    """
    Generate a set of samples from a list of polygons based on their similarity.

    Args:
        polygons (list): A list of polygons represented as lists of (x, y) tuples.
        threshold (float): The Jaccard distance threshold for creating a new sample.
            Defaults to 0.15.
        out_shape (tuple): The output shape of the rasterized polygons.
            Defaults to (720, 1280).

    Returns:
        dict: A dictionary with two keys: "index" and "Jaccard distance".
            The "index" value is the index of the polygon in the input list
            for each sample.

    """
    # Instantiate dictionary to store index and distance values
    samples = {"index": [], "Jaccard distance": []}
    idx = 0
    while idx < len(polygons) - 1:
        # Rasterize the first polygon
        first_mask = rasterize([polygons[idx]], out_shape)
        jaccard_distance = 0.0
        while jaccard_distance < threshold and idx < len(polygons) - 1:
            idx += 1
            # Rasterize the subsequent polygon
            second_mask = rasterize([polygons[idx]], out_shape)
            # Calculate Jaccard distance
            jaccard_distance = jaccard(first_mask.flatten(), second_mask.flatten())

        # Append index and distance to dictionary
        samples["index"].append(idx)
        samples["Jaccard distance"].append(jaccard_distance)
        print(f"Index: {idx}, Jaccard distance: {jaccard_distance:.4f}  ", end="\r")

    logger.info(f"Number of resulting samples: {len(samples['index'])}")

    return samples
