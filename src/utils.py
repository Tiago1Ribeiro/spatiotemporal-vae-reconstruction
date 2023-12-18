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
    # expand dimensions to be compatible with the input shape of the model
    images = np.expand_dims(images, axis=-1)

    return images


def frame_to_label(frame):
    label = np.float32(frame / 22500)
    label = np.expand_dims(label, axis=0)
    return label


def msks_paths_to_polygon_list(msks_paths, out_dim=(512, 512)):
    """
    Converts segmentation masks paths list to list of shapely multipolygons.

    Parameters:
        msks_paths {list} -- list of paths to the masks
        out_dim {tuple} -- (width, height) desired dimensions of the masks
    Returns:
        pol_list {list} -- list of shapely multipolygons
    """

    pol_list = list()
    for img_path in msks_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # image dimensions
        h, w = img.shape
        if (w, h) != out_dim:
            img = cv2.resize(img, out_dim, interpolation=cv2.INTER_CUBIC)
        polygon = mask_to_polygons(img)
        pol_list.append(polygon)
    return pol_list


def mask_to_polygons(mask_img):
    """
    Converts segmentation mask to shapely multipolygon.
    Adapted from: https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    """
    all_polygons = list()

    for shp, _ in shapes(
        source=mask_img.astype(np.uint8),
        mask=(mask_img > 0),
        transform=Affine(1.0, 0, 0, 0, 1.0, 0),
    ):
        all_polygons.append(shape(shp))

    all_polygons = MultiPolygon(all_polygons)

    # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
    # need to keep it a Multipolygon throughout
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        if all_polygons.type == "Polygon":
            all_polygons = MultiPolygon([all_polygons])

    return all_polygons


def frames2video(
    img_list,
    nome_ficheiro="video",
    fps_=25,
    titulo: str = "",
    frame_num_text=False,
    font_size: int = 1,
) -> None:
    """
    Converte lista de imagens em ficheiro AVI com a mesma resolucão da primeira
    imagem da lista.
      Parametros: - lista de imagens PNG, TIFF, JPEG, BMP, WEBP, STK, LSM ou XCF
                  - nome do ficheiro do video
      Devolve: salva vídeo no diretório de execucão
    """
    # guarda dimensões da primeira imagem
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
        filename=nome_ficheiro + ".avi",
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
        if titulo:
            cv2.putText(
                img_array[i],
                titulo,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        video.write(img_array[i])
        print(f"3. Writing frames to file {i+1}/{num_frames}", end="\r")
    video.release()
