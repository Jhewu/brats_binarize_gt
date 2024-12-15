"""
This script binarizes the ground truth data
which is numpy file with 0-3 values, by binarizing on 
values greater than 0, thus creating a mask of the whole tumor. 
This mask data is then used to train YOLOv8-seg model 
"""

"""Imports"""
import os
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""HYPERPARAMETERS"""
GT_FOLDER = "gt_data"
#GT_TO_EXTRACT = [1, 2, 3]

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def BinarizeGT(gt_slice, gt_dir, mask_dest): 
    # load the file
    slice = np.load(gt_dir)
    # binarize based on the whole tumor
    binary_mask = np.where(slice > 0, 1, 0)
    # save the mask
    gt_binary = os.path.join(mask_dest, gt_slice)
    np.save(gt_binary, binary_mask)

"""Main Runtime"""
def GTBinarizer(): 
    # set up cwd and training and validation paths
    root_dir = os.getcwd()
    gt_dir = os.path.join(root_dir, GT_FOLDER)
    gt_dir_list = os.listdir(gt_dir)

    # create destination directory
    mask_folder = "masks"
    mask_dest = os.path.join(root_dir, mask_folder)
    CreateDir(mask_dest)

    # define a thread pool executor to binarize
    # each gt_slice within gt_dir_list
    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        for gt_slice in gt_dir_list: 
            gt_slice_dir = os.path.join(gt_dir, gt_slice)
            executor.submit(BinarizeGT, gt_slice, gt_slice_dir, mask_dest)

if __name__ == "__main__": 
    GTBinarizer()
    print("\nFinish separating, please check your directory for mask\n")