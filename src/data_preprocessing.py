#!/usr/bin/env python3
"""
    -*- coding: utf-8 -*-
"""
import os
from glob import glob
import yaml
from patoolib import extract_archive
from utils import wkt2masc, load_images_from_folder


# if there isn´t unrar installed, install it with: sudo apt-get install unrar (linux)
# for windows, install it from: https://www.rarlab.com/rar_add.htm (unrarw32.exe)
if not os.path.exists("data/BurnedAreaUAV_dataset_v1.rar"):
    extract_archive("data/BurnedAreaUAV_dataset_v1.rar", program="unrar", outdir="data")

with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open(config["data"]["sampled_masks_txt"], "r", encoding="utf-8") as f:
    polygons = f.readlines()
    # extract indexes and polygons
    indexes = [int(polygon.split(",")[0]) for polygon in polygons]
    polygons = [polygon.split(",", 1)[1][:-1] for polygon in polygons]

# convert WKT files to segmentation masks : full train, sampled train and test

# create the masks folder
if not os.path.exists(config["data"]["train_dir"] + "/masks/"):
    os.makedirs(config["data"]["train_dir"] + "/masks/")
if not os.path.exists(config["data"]["test_dir"] + "/masks/"):
    os.makedirs(config["data"]["test_dir"] + "/masks/")
if not os.path.exists(config["data"]["train_sampled_dir"] + "/masks/"):
    os.makedirs(config["data"]["train_sampled_dir"] + "/masks/")

wkt2masc(
    wkt_file=config["data"]["train_wkt"],
    images_path=config["data"]["train_dir"] + "masks/",
    orig_dims=config["data"]["original_vid_dims"],
    height=config["data"]["input_size"][0],
    width=config["data"]["input_size"][1],
)

wkt2masc(
    wkt_file=config["data"]["test_wkt"],
    images_path=config["data"]["test_dir"] + "masks/",
    orig_dims=config["data"]["original_vid_dims"],
    height=config["data"]["input_size"][0],
    width=config["data"]["input_size"][1],
)

wkt2masc(
    wkt_file=config["data"]["sampled_masks_wkt"],
    images_path=config["data"]["train_sampled_dir"] + "masks/",
    orig_dims=config["data"]["original_vid_dims"],
    height=config["data"]["input_size"][0],
    width=config["data"]["input_size"][1],
)

# AQUI ---------------------------------------------
# Train directories and labels
msks_train_path = glob(os.path.join(config["data"]["train_dir"] + "masks/", "*.png"))
msks_train_basename = [os.path.basename(m) for m in msks_train_path]
msks_train_num = [int(m.split("_")[1].split(".")[0]) for m in msks_train_basename]
# mutiply masks number by 100 to get the same range as the frames
msks_train_labels = [m*100 for m in msks_train_num]

print(f"msks_train_path: {msks_train_path}")
# msks_test_path = glob(os.path.join(TEST_DIR, "*.png"))
# msks_test_basename = [os.path.basename(m) for m in msks_test_path]
# msks_test_num = [int(m.split("_")[1].split(".")[0]) for m in msks_test_basename]
# # mutiply masks number by 100 to get the same range as the frames
# msks_test_labels = [(m*100 + 20250) for m in msks_test_num]

# # For Sampled masks only
# msks_train_path = glob(os.path.join(TRAIN_SAMPLED_DIR, "*.png"))
# msks_train_labels = [m*100 for m in indexes]


# msks_test_path = glob(os.path.join(TEST_DIR, "*.png"))
# msks_test_basename = [os.path.basename(m) for m in msks_test_path]
# msks_test_num = [int(m.split("_")[1].split(".")[0]) for m in msks_test_basename]
# # mutiply masks number by 100 to get the same range as the frames
# msks_test_labels = [(m*100 + 20250) for m in msks_test_num]

# # train_imgs = load_images_from_folder(TRAIN_DIR, target_size=ARGS["IMG_SIZE"])


# # For Sampled masks only
# train_imgs = load_images_from_folder(TRAIN_SAMPLED_DIR, target_size=ARGS["IMG_SIZE"])
# test_imgs = load_images_from_folder(TEST_DIR, target_size=ARGS["IMG_SIZE"])
# train_imgs = train_imgs.reshape((-1, ARGS["IMG_SIZE"][0]*ARGS["IMG_SIZE"][1]))
# test_imgs = test_imgs.reshape((-1, ARGS["IMG_SIZE"][0]*ARGS["IMG_SIZE"][1]))
# input_shape = (ARGS["IMG_SIZE"][0]*ARGS["IMG_SIZE"][1],)

# max_val = np.max(msks_train_labels)
# train_labels = (msks_train_labels/max_val).astype(np.float32)
# train_labels = np.expand_dims(train_labels, axis=-1)

# test_labels = (msks_test_labels/max_val).astype(np.float32)
# test_labels = np.expand_dims(test_labels, axis=-1)

# print(f"train_imgs.shape: {train_imgs.shape}, train_labels.shape: {train_labels.shape}")
# print(f"test_imgs.shape: {test_imgs.shape}, test_labels.shape: {test_labels.shape}")
