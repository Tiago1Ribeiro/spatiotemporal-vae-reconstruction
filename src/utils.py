"""
Utility functions for the project.
"""
import os
import re
import numpy as np
import tensorflow as tf
from glob import glob

# import yaml
import cv2
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from log_setup import logger


def get_files_dirs_ext(dirs: list, ext: str, return_paths: bool = False):
    """
    Get all files with a specific extension in a list of directories

    Args:
        dirs (list): List of directories to search for files
        ext (str): File extension to search for
        return_paths (bool, optional): If True, returns the paths of the files. Defaults to False.

    Returns:
        list: List of files with the specific extension in the directories
    """

    result = set()

    for dir_path in dirs:
        # Use glob to find all files with the specific extension in the directory and its subdirectories
        file_paths = glob(
            os.path.join(dir_path, "**/*." + ext), recursive=True
        )

        if file_paths:
            if return_paths:
                result.update(file_paths)
            else:
                result.update([os.path.dirname(path) for path in file_paths])

    return list(result)


def draw_mask(points, orig_dims, width, height):
    """
    Draws a mask given a set of points, original dimensions, and desired dimensions.
    Parameters:
        points {list} -- list of points
        orig_dims {tuple} -- original dimensions of the masks
        width {int} -- desired width of the masks
        height {int} -- desired height of the masks
    Returns:
        A resized mask
    """
    # create empty mask
    mask = np.zeros(orig_dims, dtype=np.uint8)
    # create array with polygon points, with 2 columns (x,y)
    arr = np.array(points, dtype=np.int32).reshape((-1, 2))
    # draw mask
    cv2.drawContours(
        image=mask,
        contours=[arr],
        contourIdx=-1,
        color=(255, 255, 255),
        thickness=-1,  # if > 0, thickness of the contour; if -1, fill object
        lineType=cv2.LINE_AA,
    )
    # resize frames with Lanczos interpolation
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LANCZOS4)


def wkt_to_masc(
    wkt_file: str, images_path: str, orig_dims: Tuple[int, int], height: int, width: int
) -> None:
    """
    Converts WKT files to segmentation masks.

    Parameters:
        wkt_file {str} -- path to the WKT file
        images_path {str} -- path to the folder where the masks will be saved
        orig_dims {tuple} -- original dimensions of the masks (H, W)
        height {int} -- desired height of the masks
        width {int} -- desired width of the masks
    Returns:
        Creates PNG images of the masks
    """

    wkt = open(wkt_file, "r", encoding="utf-8")
    num_lines = len(wkt.readlines())
    cnt = 0

    logger.info(
        f"Mask Properties: Width: {width}, Height: {height}, No. of masks: {num_lines}"
    )

    pbar = tqdm(total=num_lines)

    # process each line of the WKT file
    with open(wkt_file, encoding="utf-8") as f:
        for line in f:
            # extract numbers from the line
            points = [int(s) for s in re.findall("[0-9]+", line)]
            # draw mask
            mask_resized = draw_mask(points, orig_dims, width, height)
            # save mask as PNG image
            cv2.imwrite(os.path.join(images_path, f"mask_{cnt:06d}.png"), mask_resized)
            cnt += 1
            pbar.update(1)

    pbar.close()


def wkt_to_masc(wkt_file, images_path, orig_dims, height, width):
    """
    Converts WKT files to segmentation masks.
    Parameters:
        wkt_file {str} -- path to the WKT file
        images_path {str} -- path to the folder where the masks will be saved
        orig_dims {tuple} -- original dimensions of the masks
        height {int} -- desired height of the masks
        width {int} -- desired width of the masks
    Returns:
        Creates PNG images of the masks
    """

    wkt = open(wkt_file, "r", encoding="utf-8")
    num_lines = len(wkt.readlines())
    cnt = 0

    logger.info(
        f"""
    {'-'*38}
    # \033[1mProperties of the resulting masks\033[0m
    # Width: {width}, Height: {height}
    # Number of masks to create: {num_lines}
    {'-'*38}
    """
    )

    pbar = tqdm(total=num_lines)

    # process each line of the WKT file
    with open(wkt_file, encoding="utf-8") as f:
        for line in f:
            # extract numbers from the line
            points = [int(s) for s in re.findall("[0-9]+", line)]
            # create empty mask
            mask = np.zeros(orig_dims, dtype=np.uint8)
            # create array with polygon points, with 2 columns (x,y)
            arr = np.array(points, dtype=np.int32).reshape((-1, 2))
            # draw mask
            cv2.drawContours(
                image=mask,
                contours=[arr],
                contourIdx=-1,
                color=(255, 255, 255),
                thickness=-1,  # if > 0, thickness of the contour; if -1, fill object
                lineType=cv2.LINE_AA,
            )
            # resize frames with Lanczos interpolation
            mask_resized = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
            # save mask as PNG image
            cv2.imwrite(os.path.join(images_path, f"mask_{cnt:06d}.png"), mask_resized)
            cnt += 1
            pbar.update(1)

    pbar.close()


# def load_images_from_folder(folder_path, target_size=(256, 256)):
#     images = list()
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".png"):
#             # Load the image and convert it to a Numpy array
#             # img = load_img(os.path.join(folder_path, filename), target_size)
#             img = cv2.imread(os.path.join(folder_path, filename))
#             img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
#             # img_array = np.array(img)
#             # img_array = img_to_array(img) / 255.0  # rescale pixel values to [0,1]
#             # Add the image array to the list of images
#             # images.append(img_array)
#             images.append(img)
#     # Convert the list of images to a float32 Numpy array
#     images_array = np.array(images).astype("float32")

#     return images_array


# @tf.function  # decorator to make the function callable by TensorFlow
# def load_preprocess_mask(
#     mask_path: tf.Tensor,
#     label: tf.Tensor,
#     output_dims: tf.Tensor = tf.constant((256, 256)),
#     last_frame: int = 22500,
# ) -> tf.Tensor:
#     """
#     Loads and preprocesses a mask image.
#     Parameters:
#         mask_path {tf.Tensor} -- path to the mask image
#         label {tf.Tensor} -- label of the mask
#         output_dims {tf.Tensor} -- desired dimensions of the mask
#         last_frame {int} -- last frame number
#     Returns:
#         A tuple containing the mask and the label
#     """

#     # Read and decode the image
#     mask = tf.io.read_file(mask_path)
#     mask = tf.image.decode_png(mask, channels=1)

#     # Resize the image
#     mask = tf.image.resize(mask, output_dims)

#     # Normalize the mask
#     mask = tf.math.divide(mask, 127.5) - 1

#     # Normalize the label
#     label = tf.math.divide(label, last_frame)
#     label = tf.expand_dims(label, axis=-1)
#     # Cast to float32
#     label = tf.cast(label, tf.float32)

#     return mask, label


# def load_preprocess_mask(
#     mask_path: str,
#     label: float,
#     output_dims: tuple = (256, 256),
#     last_frame: int = 22500,
# ) -> np.ndarray:
#     """
#     Loads and preprocesses a mask image.
#     Parameters:
#         mask_path {str} -- path to the mask image
#         label {float} -- label of the mask
#         output_dims {tuple} -- desired dimensions of the mask
#         last_frame {int} -- last frame number
#     Returns:
#         A tuple containing the mask and the label
#     """

#     # Check if the file exists
#     if not os.path.exists(mask_path):
#         raise FileNotFoundError(f"No such file: '{mask_path}'")

#     # Read and decode the image
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#     # Resize the image
#     mask = cv2.resize(mask, output_dims)

#     # Normalize the mask
#     mask = (mask / 127.5) - 1

#     # Normalize the label
#     label = label / last_frame
#     label = np.expand_dims(label, axis=-1)

#     return mask, label


def load_images_from_folder(
    dir: str, target_size: Tuple[int, int] = (256, 256)
) -> np.array:
    """
    Loads all images from a folder into a numpy array.
    Parameters:
        dir {str} -- path to the folder containing the images
        target_size {tuple} -- desired dimensions of the images
    Returns:
        images {numpy.ndarray} -- numpy array containing the images (N, H, W, 1)
    """

    images = []

    # Check if the folder exists
    if not os.path.exists(dir):
        print(f"Error: Folder '{dir}' not found.")
        return images

    # Loop through all files in the folder
    for filename in sorted(os.listdir(dir)):
        file_path = os.path.join(dir, filename)

        # Check if the file is a valid image
        if os.path.isfile(file_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            try:
                # Open and resize the image using OpenCV
                img = cv2.imread(file_path)
                # to grayscale => these are binary masks
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, target_size)
                # Normalize the image to [-1, 1]
                img = img.astype("float32") / 127.5 - 1
                images.append(img)
            except Exception as e:
                print(f"Error loading image '{filename}': {str(e)}")

    # Convert the list of images to a numpy array of dtype float32
    images = np.array(images).astype("float32")
    # Expand dimensions to be compatible with the input shape of the model
    images = np.expand_dims(images, axis=-1)

    return images


def frame_to_label(frame: int, max_frame: int = 22500) -> np.array:
    """
    Converts the absolute frame number to a 0 to 1 value float32.

    Parameters:
    frame (int): The absolute frame number.
    max_frame (int): The maximum frame number. Default is 22500.

    Returns:
    np.array: The normalized frame number as a 0 to 1 value float32.
    """
    # Convert the frame number to a 0 to 1 value float32
    label = np.float32(frame / max_frame)
    # Expand the dimensions of the label to make it compatible with the model
    label = np.expand_dims(label, axis=0)
    return label


def frames_to_video(
    img_list_dir: str,
    output_dir: str = "",
    output_resolution: Tuple[int, int] = (800, 600),
    file_name: str = "video",
    f_ps: int = 25,
    title: str = "",
    frame_num_text: bool = False,
    font_size: int = 1,
) -> None:
    """
    Converts a list of images into an MP4 file with the same resolution as the first image in the list.

    Parameters:
        img_list_dir (str): The path to the directory containing the images.
        output_dir (str): The path to the directory where the video will be saved. Default is the current directory.
        output_resolution (tuple): The resolution of the output video. Default is (800, 600).
        file_name (str): The name of the video file. Default is "video".
        f_ps (int): The frames per second. Default is 25.
        title (str): The title of the video. Default is an empty string.
        frame_num_text (bool): Whether to add frame numbers to the video. Default is False.
        font_size (int): The font size of the frame numbers. Default is 1.

    Side effects:
        Saves the MP4 video in the specified output directory.

    Example Usage:
        frames_to_video(img_list_dir="path/to/images", output_dir="path/to/output", file_name="output_video", output_resolution=(800, 600))
    """

    logger.info("Creating image list...                          ")
    img_list = sorted(
        [
            os.path.join(img_list_dir, filename)
            for filename in os.listdir(img_list_dir)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    # Check if the image files exist and are valid images
    for img_file in img_list:
        if not os.path.isfile(img_file):
            logger.error(f"File {img_file} does not exist.")
            continue
        img = cv2.imread(img_file)
        if img is None:
            logger.error(f"File {img_file} is not a valid image.")
            continue

    # Save the dimensions of the first image
    img = cv2.imread(img_list[0])
    height, width, _ = img.shape
    input_size = (width, height)
    num_frames = len(img_list)

    output_path = os.path.join(output_dir, file_name + ".mp4")

    video = cv2.VideoWriter(
        filename=output_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=f_ps,
        frameSize=output_resolution,
    )
    try:
        for i, img_file in enumerate(img_list):
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if output_resolution != input_size:
                img = cv2.resize(img, output_resolution, cv2.INTER_LANCZOS4)
            if frame_num_text:
                frame_number_text = f"Frame: {i:06d}/{num_frames:06d}"
                cv2.putText(
                    img,
                    frame_number_text,
                    (50, 100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size,
                    (255, 200, 200),
                    2,
                    cv2.LINE_AA,
                )
            if title:
                cv2.putText(
                    img,
                    title,
                    (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size,
                    (255, 200, 200),
                    2,
                    cv2.LINE_AA,
                )
            video.write(img)
            if i % 1000 == 0:
                logger.info(f"Writing frames to file {i+1}/{num_frames}")
    finally:
        video.release()
        logger.info(f"Saved video to {output_path}")


def frame_to_label(frame, max_frame=22500):
    """
    Label encoding function. It converts the frame number to a 0 to 1 value float32.

    Parameters:
    frame (int): The absolute frame number.
    max_frame (int): The maximum frame number. Default is 22500.

    Returns:
    np.array: The normalized frame number as a 0 to 1 value float32.
    """
    # Convert the frame number to a 0 to 1 value float32
    label = np.float32(frame / max_frame)
    # Expand the dimensions of the label to make it compatible with the model
    label = np.expand_dims(label, axis=0)
    return label


def label_to_frame(label, max_frame=22500):
    """
    Label decoding function. It converts the normalized frame number to an absolute frame number.

    Parameters:
    label (np.array): The normalized frame number as a 0 to 1 value float32.
    max_frame (int): The maximum frame number. Default is 22500.

    Returns:
    int: The absolute frame number.
    """
    # Convert the label to an absolute frame number
    frame = int(label * max_frame)
    return frame


def save_history(history, path):
    """
    Save the Keras training history to CSV file

    Parameters:
    - history (History): The Keras History object to save.
    - path (str): The path where the history should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert the history object to a dictionary
    history_dict = dict()
    for key in history.history.keys():
        history_dict[key] = history.history[key]

    # Save the history to a CSV file
    df = pd.DataFrame(history_dict)
    df.to_csv(path, index=False)
    logger.info(f"Saved history to {path}")
