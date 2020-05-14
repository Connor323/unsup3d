###################################################################################################
# Script for parsing the lip dataset into cropped images
# Author: Hans
# Creating date: 05/14/2020
###################################################################################################

import argparse
import cv2
import numpy as np
import glob
import os
from os import path as osp
from tqdm import tqdm

def get_arguments(): 
    parser = argparse.ArgumentParser("Building the cropped LIP dataset") 
    parser.add_argument('--dataroot', type=str, default="/pub1/hao66/dataset/lip_dataset", help='path to thce dataset')
    parser.add_argument('--results_dir', type=str, default="/pub1/hao66/dataset/lip_dataset_cropped", help='path to thce result directory')
    return parser.parse_args()

def get_images(dataroot):
    """Get training and testing images.
    Params: 
        dataroot: <str> dataset root path
    Returns:
        train_files: <list> a list of training file paths
        test_files: <list> a list of testing file paths
    """
    root = osp.join(dataroot, 'TrainVal_images')
    # load training files
    path = osp.join(root, 'train_images')
    train_files = sorted(glob.glob(osp.join(path, "*.jpg")))
    # load testing files
    path = osp.join(root, 'val_images')
    test_files = sorted(glob.glob(osp.join(path, "*.jpg")))
    return train_files, test_files

def get_annotations(dataroot):
    """Get training and testing body-part annotations.
    Params: 
        dataroot: <str> dataset root path
    Returns:
        train_anno_files: <list> a list of training file paths
        test_anno_files: <list> a list of testing file paths
    """
    root = osp.join(dataroot, 'TrainVal_parsing_annotations')
    # load training files
    path = osp.join(root, 'train_segmentations')
    train_anno_files = sorted(glob.glob(osp.join(path, "*.png")))
    # load testing files
    path = osp.join(root, 'val_segmentations')
    test_anno_files = sorted(glob.glob(osp.join(path, "*.png")))
    return train_anno_files, test_anno_files

def check_image_annotation_consistency(*args):
    assert len(args) % 2 == 0, "number of input should be even, but got %d" % len(args)
    image_file_list = args[:len(args)//2]
    anno_file_list = args[len(args)//2:]
    for image_files, anno_files in zip(image_file_list, anno_file_list):
        assert len(image_files) == len(anno_files) != 0, "%d != %d" % (len(image_files), len(anno_files))
        for image_file, anno_file in zip(image_files, anno_files):
            image_name = osp.splitext(osp.basename(image_file))[0]
            anno_name = osp.splitext(osp.basename(anno_file))[0]
            assert image_name == anno_name, "%s != %s" % (image_name, anno_name)

def crop_images(image_files, anno_files):
    """Save the images to the result path given the image paths.
    Params: 
        image_files: <list> a list of image file paths
        anno_files: <list> a list of annoataion image file paths
    Returns:
        crop_images: <list> a list of [numpy.array, image name, label] of cropped images
        crop_annos: <list> a list of [numpy.array, image name, label] of cropped annotation images
    """
    crop_images, crop_annos = [], []
    for image_file, anno_file in tqdm(zip(image_files, anno_files), total=len(image_files)):
        image = cv2.imread(image_file)
        anno = cv2.imread(anno_file, 0)
        image_name = osp.splitext(osp.basename(image_file))[0]
        labels = np.unique(anno)
        for label in labels:
            ys, xs = np.where(anno != label)
            if not len(xs) or not len(ys): continue
            bbox = [xs.min(), ys.min(), xs.max(), ys.max()] # [left, top, right, bottom]
            patch_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            patch_anno = anno[bbox[1]:bbox[3], bbox[0]:bbox[2]] == label
            crop_images.append([patch_image, image_name, label])
            crop_annos.append([patch_anno, image_name, label])
    return crop_images, crop_annos

def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)

def save_images(images, annos, results_dir):
    """Save the images to the result path given the image paths.
    Params: 
        images: <list> a list of [numpy.array, image name, label] of cropped images
        annos: <list> a list of [numpy.array, image name, label] of cropped annotation images
        results_dir: <str> root path for the small image
    Returns:
        None 
    """
    print("Saving images...")
    mkdir(results_dir)
    mkdir(osp.join(results_dir, "images"))
    mkdir(osp.join(results_dir, "annotations"))
    for image, name, label in tqdm(images):
        save_name = name + "-" + str(label) + ".png"
        cv2.imwrite(osp.join(results_dir, "images", save_name), image)
    for anno, name, label in tqdm(annos):
        save_name = name + "-" + str(label) + ".png"
        cv2.imwrite(osp.join(results_dir, "annotations", save_name), (anno * 255).astype(np.uint8))

if __name__ == '__main__':
    opt = get_arguments()
    train_files, test_files = get_images(opt.dataroot)
    train_anno_files, test_anno_files = get_annotations(opt.dataroot)
    check_image_annotation_consistency(train_files, test_files, train_anno_files, test_anno_files)
    
    train_crop_images, train_crop_masks = crop_images(train_files, train_anno_files)
    test_crop_images, test_crop_masks = crop_images(test_files, test_anno_files)
    
    
    mkdir(opt.results_dir)
    save_images(train_crop_images, train_crop_masks, osp.join(opt.results_dir, "train"))
    save_images(test_crop_images, test_crop_masks, osp.join(opt.results_dir, "test"))