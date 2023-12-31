#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
from rasterio.features import shapes
from rasterio import Affine
from shapely.geometry import shape, MultiPolygon
from log_setup import logger


# def mask_to_poly(mask_img: np.ndarray) -> MultiPolygon:
#     """
#     Converts a segmentation mask to a shapely multipolygon.

#     Parameters:
#     mask_img (numpy.ndarray): The segmentation mask.

#     Returns:
#     shapely.geometry.MultiPolygon: The shapely multipolygon.
#     """
#     all_polygons = list()

#     for shp, _ in shapes(
#         source=mask_img.astype(np.uint8),
#         mask=(mask_img > 0),
#         transform=Affine(1.0, 0, 0, 0, 1.0, 0),
#     ):
#         all_polygons.append(shape(shp))

#     all_polygons = MultiPolygon(all_polygons)

#     if not all_polygons.is_valid:
#         all_polygons = all_polygons.buffer(0)
#         if all_polygons.geom_type == "Polygon":
#             all_polygons = MultiPolygon([all_polygons])

#     return all_polygons


# def save_polygons_to_wkt(polygon_list: list, file_path: str) -> None:
#     """
#     Save a list of shapely polygons to a WKT format file.

#     Parameters:
#     polygon_list (list): List of shapely polygons.
#     file_path (str): Path to the output file.
#     """
#     with open(file_path, "w") as f:
#         for polygon in polygon_list:
#             f.write(polygon.wkt + "\n")


# def masks_to_polygons(
#     msks_paths: list, out_dim: tuple = (512, 512), save_path: str = None
# ) -> list:
#     """
#     Converts a list of paths to segmentation masks into a list of shapely multipolygons.

#     Parameters:
#     msks_paths (list): A list of paths to the masks.
#     out_dim (tuple): The desired dimensions of the masks. Default is (512, 512).
#     save_path (str): Optional. If provided, the function saves the polygons to a WKT file.

#     Returns:
#     list: A list of shapely multipolygons.
#     """
#     start_time = time.time()

#     logger.info("Converting masks to polygons...")

#     pol_list = list()
#     i = 0
#     for img_path in msks_paths:
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         h, w = img.shape
#         if (w, h) != out_dim:
#             img = cv2.resize(img, out_dim, interpolation=cv2.INTER_CUBIC)
#         polygon = mask_to_poly(img)
#         pol_list.append(polygon)
#         i += 1

#         elapsed_time = time.time() - start_time
#         print(
#             f"Processed {i} masks out of {len(msks_paths)} | "
#             f"Time elapsed: {elapsed_time:.2f}s  ",
#             end="\r",
#         )
#     if save_path:
#         save_polygons_to_wkt(pol_list, save_path)
#         logger.info(f"Saved polygons to {save_path}")

#     return pol_list


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
) -> None:
    """
    Converts a list of paths to segmentation masks into a list of shapely multipolygons.

    Parameters:
    msks_paths (list): A list of paths to the masks.
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
