"""
Utility functions for the project.
"""
import os
import re
import numpy as np

# from glob import glob
# import yaml
import cv2
from tqdm import tqdm
from typing import List


# def load_config(config_path):
#     """
#     Loads a YAML configuration file.
#     Parameters:
#         config_path {str} -- path to the YAML configuration file
#     Returns:
#         config {dict} -- dictionary with the configuration parameters
#     """
#     with open(config_path, "r", encoding="utf-8") as stream:
#         try:
#             config = yaml.safe_load(stream)
#             return config
#         except yaml.YAMLError as exc:
#             print(exc)


def wkt2masc(wkt_file, images_path, orig_dims, height, width):
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

    print(
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


def load_images_from_folder(folder_path, target_size=(256, 256)):
    """
    Loads all images from a folder into a numpy array.
    Parameters:
        folder_path {str} -- path to the folder containing the images
        target_size {tuple} -- desired dimensions of the images
    Returns:
        images {numpy.ndarray} -- numpy array containing the images (N, H, W, 1)
    """

    images = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return images

    # Loop through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

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


def frames2video(
    img_list: List[str],
    file_name: str = "video",
    fps_: int = 25,
    title: str = "",
    frame_num_text: bool = False,
    font_size: int = 1,
) -> None:
    """
    Converts a list of images into an AVI file with the same resolution as the first image in the list.

    Parameters:
    img_list (List[str]): A list of image file paths.
    nome_ficheiro (str): The name of the video file. Default is "video".
    fps_ (int): The frames per second. Default is 25.
    title (str): The title of the video. Default is an empty string.
    frame_num_text (bool): Whether to add frame numbers to the video. Default is False.
    font_size (int): The font size of the frame numbers. Default is 1.

    Returns:
    None

    Side effects:
    Saves the video in the current directory.
    """
    # Check if the image files exist and are valid images
    for img_file in img_list:
        if not os.path.isfile(img_file):
            raise ValueError(f"File {img_file} does not exist.")
        img = cv2.imread(img_file)
        if img is None:
            raise ValueError(f"File {img_file} is not a valid image.")

    # Save the dimensions of the first image
    img = cv2.imread(img_list[0])
    height, width, _ = img.shape
    size = (width, height)
    num_frames = len(img_list)

    img_array = list()
    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img_array.append(img)
        print(f"1. Appending frames {i+1}/{num_frames}", end="\r")

    print("2. Creating video writer...")
    video = cv2.VideoWriter(
        filename=file_name + ".avi",
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps_,
        frameSize=size,
    )

    for i in range(len(img_array)):
        if frame_num_text:
            frame_number_text = f"frame_{i:06d}"
            cv2.putText(
                img_array[i],
                frame_number_text,
                (width - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 100, 100),
                2,
                cv2.LINE_AA,
            )
        if title:
            cv2.putText(
                img_array[i],
                title,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        video.write(img_array[i])
        print(f"3. Writing frames to file {i+1}/{num_frames}    ", end="\r")
    video.release()


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
