#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Function for generating geometric affine augmentations in images, while also
handling bounding boxes.

@author: vasileios vonikakis
"""

import io
import imageio
import boto3
import copy
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import matplotlib.cm as cmx
from skimage.transform import warp, AffineTransform
from skimage.exposure import equalize_adapthist
from skimage.util import random_noise
from skimage.exposure import adjust_log
from skimage.color import rgb2gray
from skimage.util import img_as_float




def adjust_image_colorfulness(image, degree):
    # adjusts the colorfulness of an image according to the degree
    
    image = img_as_float(image)  # [0,1]
    image_gray = rgb2gray(image)  # [0,1]
    image_gray = np.dstack((image_gray, image_gray, image_gray))
    
    image_color_delta = image - image_gray  # deviations from gray
    
    image_adjusted = image_gray
    image_adjusted = image_adjusted + image_color_delta * degree
    
    image_adjusted[image_adjusted>1] = 1
    image_adjusted[image_adjusted<0] = 0
    
    return image_adjusted



def cast_image_as_uint8(image):
    # converts a float image of [0,1] to an uint8 image of [0,255]
    
    image *= 255
    image[image>255] = 255
    image[image<0] = 0

    return image.astype(np.uint8)
    

    

def compress_image(image, compression_level):
    # Applies JPEG compression on an image's numpy array
    
    # Create a Pillow Image object from the input NumPy array
    img = Image.fromarray(cast_image_as_uint8(image))

    # Set the JPEG quality parameter based on the compression level (0-100)
    if compression_level > 100: compression_level = 100
    elif compression_level < 0: compression_level = 0
    quality = int(100 - compression_level)

    # Compress the image using JPEG algorithm
    output_buffer = io.BytesIO()
    img.save(output_buffer, format='JPEG', quality=quality)
    compressed_img = Image.open(output_buffer)
    
    # Convert the compressed image back to a NumPy array and return it
    return img_as_float(np.array(compressed_img, dtype=np.uint8))  # in [0,1]
    


def convert_image_temperature(image, degree):
    # adjusts the color temperature of an image based on the degree
    
    image = img_as_float(image)  # [0,1]
    
    DEGREE_MIN = -1
    DEGREE_MAX = 1
    
    R_COOL = 150
    G_COOL = 200
    B_COOL = 255
    
    R_WARM = 255
    G_WARM = 200
    B_WARM = 50

    if degree < 0:
        r = R_COOL + ((255 - R_COOL) / (0 - DEGREE_MIN)) * (degree - DEGREE_MIN)
        g = G_COOL + ((255 - G_COOL) / (0 - DEGREE_MIN)) * (degree - DEGREE_MIN)
        b = B_COOL + ((255 - B_COOL) / (0 - DEGREE_MIN)) * (degree - DEGREE_MIN)
    else:
        r = R_WARM + ((255 - R_WARM) / (DEGREE_MAX - 0)) * (DEGREE_MAX - degree)
        g = G_WARM + ((255 - G_WARM) / (DEGREE_MAX - 0)) * (DEGREE_MAX - degree)
        b = B_WARM + ((255 - B_WARM) / (DEGREE_MAX - 0)) * (DEGREE_MAX - degree)
    
    image[:,:,0] *= r / 255.0
    image[:,:,1] *= g / 255.0
    image[:,:,2] *= b / 255.0
    
    image[image>1] = 1
    image[image<0] = 0
    
    return image




def flip_image(image, direction='lr'):
    # direction='lr' -> flip left right
    # direction='ud' -> flip up down

    image_flipped = image.copy()
    
    if direction == 'ud':
        image_flipped = np.flipud(image_flipped)
    else:
        image_flipped = np.fliplr(image_flipped)

    return image_flipped



def flip_bboxes(ls_bboxes, image_width, image_height, direction='lr'):
    # direction='lr' -> flip left right
    # direction='ud' -> flip up down
        
    ls_bboxes_flipped = []
    
    for bbox in ls_bboxes:
        
        if direction == 'ud':
            ls_bboxes_flipped.append(
                {
                    'left': bbox['left'],
                    'top': image_height - (bbox['top'] + bbox['height']),  # height-y_down_right
                    'width': bbox['width'], 
                    'height': bbox['height'],
                    'class_id': bbox['class_id']
                }
            )
        else:
            ls_bboxes_flipped.append(
                {
                    'left': image_width - (bbox['left'] + bbox['width']),  # width-x_down_right
                    'top': bbox['top'], 
                    'width': bbox['width'], 
                    'height': bbox['height'],
                    'class_id': bbox['class_id']
                }
            )
    
    return ls_bboxes_flipped






def visualize_image(
    image, 
    title=None, 
    bboxes=None, 
    max_number_of_classes=None,
    display = True
    ):
    """
    ---------------------------------------------------------------------------
           Visualizes an image with or without bounding box detections
    ---------------------------------------------------------------------------
    
    INPUTS
    ------
    image: RGB numpy array 
        Image array that will be visualized. 
    title: string or None
        The title that will be depicted on top of the image.    
    bboxes: list of dictionaries or None
        The list of the given bounding boxes in the image. A bounding box is 
        defined by 5 numbers of a dictionary: {"class_id", "top", "left", 
        "height", "width"}. For example, a possbile bounding box could be:
        {"class_id": 0, "top": 44, "left": 394, "height": 189, "width": 147}
        Top, left, height and width are defined in pixels.
    max_number_of_classes: int or None
        The maximum number of object classes. This is needed in order to have
        a fixed color for each bounding box class when visualizing results. 

    """
    
    # show the image
    if display is True:
        plt.figure()
        
    plt.imshow(
        image, 
        vmin=0, 
        vmax=255, 
        interpolation='bilinear'
    )
    if title is not None:
        if display is True:
            plt.title(title)
        else:
            plt.title(title, loc='left')
    plt.axis('off')
    plt.tight_layout(True)
    
    # show the bounding boxes (if they are provided)
    if bboxes is not None:
        # set fixed colors per class using the jet color map
        cm = plt.get_cmap('jet') 
        if max_number_of_classes is None:
            max_number_of_classes = len(set([ bbox['class_id'] for bbox in bboxes ]))
        cNorm  = colors.Normalize(vmin=0, vmax=max_number_of_classes-1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        
        for bbox in bboxes:
            cls_id = int(bbox['class_id'])
            color_val = scalarMap.to_rgba(cls_id)[0:3]  # keep only rgb, discard a
            plt.gca().add_patch(  # depict the rectangles
                Rectangle(
                    xy=(bbox['left'],bbox['top']),
                    width=bbox['width'],
                    height=bbox['height'],
                    linewidth=2,
                    edgecolor=color_val,
                    facecolor='none'
                    )
                )
            plt.gca().text(  # depict the class names
                bbox['left'],
                bbox['top'] - 2,
                "{:s}".format(str(bbox['class_id'])),
                bbox=dict(facecolor=color_val, alpha=0.5),
                fontsize=8,
                color="white",
            )
    
    if display is True: 
        plt.show()










def augment_image(
        image,
        bboxes = None,
        max_number_of_classes=None,
        how_many=1,
        random_seed=None,
        range_scale=None, 
        range_translation=None,
        range_rotation=None,
        range_shear=None,
        range_noise=None,
        range_brightness=None,
        range_colorfulness=None,
        range_color_temperature=None,
        range_compression=None,
        flip_lr = None,
        flip_ud = None,
        enhance = None,
        bbox_truncate = True,
        bbox_discard_thr = 0.85,
        display=False,
        verbose=False
        ):
    
    '''
    ---------------------------------------------------------------------------
          Functiont that generates random augmentations for a given image
    ---------------------------------------------------------------------------
    The function apply affine distortions on a given image and generates many
    random variations of it. If bounding boxes are provided, then they are also
    transformed to the new distorted image and returned back. The function is
    based on the scikit-image library.
    
    INPUTS
    ------
    image: numpy array or string
        If string, then it represents a local image filename that the function
        will try to open. If numpy array, then the function will use its values 
        directly. The image can be either color (RGB) or grayscale. If 
        grayscale, then the image will be converted to RGB. 
    bboxes: list of dictionaries or None
        The list of the given bounding boxes in the image. A bounding box is 
        defined by 5 numbers of a dictionary: {"class_id", "top", "left", 
        "height", "width"}. For example, a possbile bounding box could be:
        {"class_id": 0, "top": 44, "left": 394, "height": 189, "width": 147}
        Top, left, height and width are defined in pixels.
    max_number_of_classes: int or None
        The maximum number of object classes. This is needed in order to have
        a fixed color for each bounding box class when visualizing results. 
        If None, then the colors of bounding boxes will be different across
        images. 
    how_many: int
        How many augmentations to generate per input image.
    random_seed: int
        Number for setting the random number generator, in order to control 
        reproducibility.
    range_scale: (min, max) tuple of float, or None
        Minimum and maximum range of possible zooming/unzooming scale factors. 
        Value range: float[0.1, 5], <1=zoom in, >1=zoom out, e.g. (0.5,1.5). 
        If None, then this transformation is deactivated. 
    range_translation: (min, max) tuple of integer, or None
        Minimum and maximum range for offseting the (x,y) position of the image 
        (in pixels). Value range: int[0,image_min_dim], where image_min_dim is 
        the smallest dimension between width and hight. E.g. value (-100, 100). 
        If None, then this transformation is deactivated. 
    range_rotation: (min, max) tuple of float, or None
        Minimum and maximum range for rotating image left/right (in degrees). 
        Value range: float[-360,360], e.g.(-45, 45).
        If None, then this transformation is deactivated.
    range_shear: (min, max) tuple of integer, or None
        Minimum and maximum range for skewing image left/right (in degrees). 
        Value range: float[-360,360], e.g.(-45, 45).
        If None, then this transformation is deactivated.
    range_noise: (min, max) tuple of float, or None
        Minimum and maximum range of noise variance. 
        Value range: float[0,3], e.g. (0, 0.001). 
        If None, then this transformation is deactivated. 
    range_brightness: (min, max) tuple of float, or None
        Minimum and maximum range for brightness gain. 
        Value range: float[0.1,10], 1=no change, <1=darken, >1=brighten, 
        e.g. (0.5, 1.5). If None, then this transformation is deactivated. 
    range_colorfulness: (min, max) tuple of float, or None
        Minimum and maximum range for color saturation. 
        Value range: float[0,5], 1=no change, 0=grayscale, >1=more saturated, 
        e.g. (0.5, 1.5). If None, then this transformation is deactivated. 
    range_color_temperature: (min, max) tuple of float, or None
        Minimum and maximum range for color temperature (cool/warm). 
        Value range: float[-1,1], -1=cool, 0=no change, 1=warm, 
        e.g. (-0.5, 1.0). If None, then this transformation is deactivated. 
    range_compression: (min, max) tuple of int, or None
        Minimum and maximum range for jpeg compression. 
        Value range: int[0,100], 0=no compression (best quality), 
        100=max compression (worst quality) 
        e.g. (10, 80). If None, then this transformation is deactivated. 
    flip_lr: string or None
        None: no left-right flipping is applied.
        'all': all images are flipped left-to-right and the original version 
            is also kept (doubles the number of images).
        'random': images are flipped left-to-right randomly (same number of 
            total images).
    flip_ud: string or None
        None: no up-down flipping is applied.
        'all': all images are flipped up-to-down and the original version 
            is also kept (doubles the number of images).
        'random': images are flipped up-to-down randomly (same number of 
            total images).
    enhance: string or None
        None: no image enhancement is applied.
        'all': all images are enhanced and the original version is also 
            kept (doubles the number of images).
        'random': images are enhanced randomly (same number of total images).
    bbox_truncate: bool
        Whether or not to truncate bounding boxes within image boundaries.
    bbox_discard_thr: float [0,1]
        Helps to discard any new bounding boxes that are located mostly 
        outside the image boundaries, due to the augmentations. If the ratio 
        of the surface of a new bounding box (after image augmentation), over 
        the surface of the original bounding box, is less than 
        bbox_discard_thr, then the new bounding box is discarded (i.e. object 
        lies mostly outside the new image). Values closer to 1 are more strict
        whereas values closer to 0 are more permissive. 
    display: string or True / False
        Show visualizations and details or not.
        False: no visualizations.
        True: display images individually.
        'single': display images individually.
        'grid': display up to 10 images on a grid.
    verbose: boolean
        Show or not warnings and other messages. 
    
    OUTPUT
    ------
    dcionary containing:
        Augmented images.
        Transformed bounding boxes.
        Details about each augmentation. 
        List of discarded bounding boxes (being outside the image).
        
    '''
    
    # resolve input image
    
    if type(image) is str:
        image_filename = image
        image = imageio.imread(image_filename)  # load image
        
    image_min_dim = min(image.shape[0], image.shape[1])
    if len(image.shape) == 2:
        image = np.dstack((image, image, image))  # grayscale -> RGB
        
    if image.shape[2] > 3:  # if PNG (R,G,B,A)
        image = np.dstack((image[:,:,0], image[:,:,1], image[:,:,2]))  # keep only RGB
        
    
    # print(image.shape)
    
    #------------------------------------------------- sanity check for ranges
    
    # value ranges
    RANGE_SCALE_MIN = 0.1
    RANGE_SCALE_MAX = 5
    RANGE_TRANSLATION_MIN = -image_min_dim
    RANGE_TRANSLATION_MAX = image_min_dim
    RANGE_ROTATION_MIN = -360
    RANGE_ROTATION_MAX = 360
    RANGE_shear_MIN = -360
    RANGE_shear_MAX = 360
    RANGE_NOISE_MIN = 0.0
    RANGE_NOISE_MAX = 3
    RANGE_BRIGHTNESS_MIN = 0.1
    RANGE_BRIGHTNESS_MAX = 10
    RANGE_colorfulness_MIN = 0.0
    RANGE_colorfulness_MAX = 5
    RANGE_COLOR_TEMPERATURE_MIN = -1
    RANGE_COLOR_TEMPERATURE_MAX = 1
    RANGE_COMPRESSION_MIN = 0
    RANGE_COMPRESSION_MAX = 100
    BBOX_DISCARD_THR_MIN = 0
    BBOX_DISCARD_THR_MAX = 1
    BBOX_DISCARD_THR_DEFAULT = 0.85
    
    param = {
        'range_scale': {'flag':0, 'min':RANGE_SCALE_MIN, 'max':RANGE_SCALE_MAX},
        'range_translation': {'flag':0, 'min':RANGE_TRANSLATION_MIN, 'max':RANGE_TRANSLATION_MAX},
        'range_rotation': {'flag':0, 'min':RANGE_ROTATION_MIN, 'max':RANGE_ROTATION_MAX},
        'range_shear': {'flag':0, 'min':RANGE_shear_MIN, 'max':RANGE_shear_MAX},
        'range_noise': {'flag':0, 'min':RANGE_NOISE_MIN, 'max':RANGE_NOISE_MAX},
        'range_brightness': {'flag':0, 'min':RANGE_BRIGHTNESS_MIN, 'max':RANGE_BRIGHTNESS_MAX},
        'range_colorfulness': {'flag':0, 'min':RANGE_colorfulness_MIN, 'max':RANGE_colorfulness_MAX},
        'range_color_temperature': {'flag':0, 'min':RANGE_COLOR_TEMPERATURE_MIN, 'max':RANGE_COLOR_TEMPERATURE_MAX},
        'range_compression': {'flag':0, 'min':RANGE_COMPRESSION_MIN, 'max':RANGE_COMPRESSION_MAX},
        'flip_lr': {'flag':0},
        'flip_ud': {'flag':0},
        'enhance': {'flag':0},
        'bbox_discard_thr': {'flag':0},
        'bboxes': {'flag':0}
    }
        
    # checking range_scale
    if type(range_scale) is tuple:
        if len(range_scale) != 2: param['range_scale']['flag'] = 1
        else:
            if range_scale[1] <= range_scale[0]:
                range_scale = (range_scale[1], range_scale[0])
            if range_scale[0] < RANGE_SCALE_MIN:
                range_scale = (RANGE_SCALE_MIN, range_scale[1])
                param['range_scale']['flag'] = 2
            if range_scale[1] > RANGE_SCALE_MAX:
                range_scale = (range_scale[0], RANGE_SCALE_MAX)
                param['range_scale']['flag'] = 2
    elif range_scale is not None: param['range_scale']['flag'] = 1
    
    # checking range_translation
    if type(range_translation) is tuple:
        if len(range_translation) != 2: param['range_translation']['flag'] = 1
        else:
            if range_translation[1] <= range_translation[0]:
                range_translation = (range_translation[1], range_translation[0])
            if range_translation[0] < RANGE_TRANSLATION_MIN:
                range_translation = (RANGE_TRANSLATION_MIN, range_translation[1])
                param['range_translation']['flag'] = 2
            if range_translation[1] > RANGE_TRANSLATION_MAX:
                range_translation = (range_translation[0], RANGE_TRANSLATION_MAX)
                param['range_translation']['flag'] = 2
    elif range_translation is not None: param['range_translation']['flag'] = 1
    
    # checking range_rotation
    if type(range_rotation) is tuple:
        if len(range_rotation) != 2: param['range_rotation']['flag'] = 1
        else:
            if range_rotation[1] <= range_rotation[0]:
                range_rotation = (range_rotation[1], range_rotation[0])
            if range_rotation[0] < RANGE_ROTATION_MIN:
                range_rotation = (RANGE_ROTATION_MIN, range_rotation[1])
                param['range_rotation']['flag'] = 2
            if range_rotation[1] > RANGE_ROTATION_MAX:
                range_rotation = (range_rotation[0], RANGE_ROTATION_MAX)
                param['range_rotation']['flag'] = 2
    elif range_rotation is not None: param['range_rotation']['flag'] = 1
    
    # checking range_shear
    if type(range_shear) is tuple:
        if len(range_shear) != 2: param['range_shear']['flag'] = 1
        else:
            if range_shear[1] <= range_shear[0]:
                range_shear = (range_shear[1], range_shear[0])
            if range_shear[0] < RANGE_shear_MIN:
                range_shear = (RANGE_shear_MIN, range_shear[1])
                param['range_shear']['flag'] = 2
            if range_shear[1] > RANGE_shear_MAX:
                range_shear = (range_shear[0], RANGE_shear_MAX)
                param['range_shear']['flag'] = 2
    elif range_shear is not None: param['range_shear']['flag'] = 1
    
    # checking range_noise
    if type(range_noise) is tuple:
        if len(range_noise) != 2: param['range_noise']['flag'] = 1
        else:
            if range_noise[1] <= range_noise[0]:
                range_noise = (range_noise[1], range_noise[0])
            if range_noise[0] < RANGE_NOISE_MIN:
                range_noise = (RANGE_NOISE_MIN, range_noise[1])
                param['range_noise']['flag'] = 2
            if range_noise[1] > RANGE_NOISE_MAX:
                range_noise = (range_noise[0], RANGE_NOISE_MAX)
                param['range_noise']['flag'] = 2
    elif range_noise is not None: param['range_noise']['flag'] = 1
    
    # checking range_brightness
    if type(range_brightness) is tuple:
        if len(range_brightness) != 2: param['range_brightness']['flag'] = 1
        else:
            if range_brightness[1] <= range_brightness[0]:
                range_brightness = (range_brightness[1], range_brightness[0])
            if range_brightness[0] < RANGE_BRIGHTNESS_MIN:
                range_brightness = (RANGE_BRIGHTNESS_MIN, range_brightness[1])
                param['range_brightness']['flag'] = 2
            if range_brightness[1] > RANGE_BRIGHTNESS_MAX:
                range_brightness = (range_brightness[0], RANGE_BRIGHTNESS_MAX)
                param['range_brightness']['flag'] = 2
    elif range_brightness is not None: param['range_brightness']['flag'] = 1
    
    # checking range_colorfulness
    if type(range_colorfulness) is tuple:
        if len(range_colorfulness) != 2: param['range_colorfulness']['flag'] = 1
        else:
            if range_colorfulness[1] <= range_colorfulness[0]:
                range_colorfulness = (range_colorfulness[1], range_colorfulness[0])
            if range_colorfulness[0] < RANGE_colorfulness_MIN:
                range_colorfulness = (RANGE_colorfulness_MIN, range_colorfulness[1])
                param['range_colorfulness']['flag'] = 2
            if range_colorfulness[1] > RANGE_colorfulness_MAX:
                range_colorfulness = (range_colorfulness[0], RANGE_colorfulness_MAX)
                param['range_colorfulness']['flag'] = 2
    elif range_colorfulness is not None: param['range_colorfulness']['flag'] = 1
    
    # checking range_color_temperature
    if type(range_color_temperature) is tuple:
        if len(range_color_temperature) != 2: param['range_color_temperature']['flag'] = 1
        else:
            if range_color_temperature[1] <= range_color_temperature[0]:
                range_color_temperature = (range_color_temperature[1], range_color_temperature[0])
            if range_color_temperature[0] < RANGE_COLOR_TEMPERATURE_MIN:
                range_color_temperature = (RANGE_COLOR_TEMPERATURE_MIN, range_color_temperature[1])
                param['range_color_temperature']['flag'] = 2
            if range_color_temperature[1] > RANGE_COLOR_TEMPERATURE_MAX:
                range_color_temperature = (range_color_temperature[0], RANGE_COLOR_TEMPERATURE_MAX)
                param['range_color_temperature']['flag'] = 2
    elif range_color_temperature is not None: param['range_color_temperature']['flag'] = 1
    
    # checking range_compression
    if type(range_compression) is tuple:
        if len(range_compression) != 2: param['range_compression']['flag'] = 1
        else:
            if range_compression[1] <= range_compression[0]:
                range_compression = (range_compression[1], range_compression[0])
            if range_compression[0] < RANGE_COMPRESSION_MIN:
                range_compression = (RANGE_COMPRESSION_MIN, range_compression[1])
                param['range_compression']['flag'] = 2
            if range_compression[1] > RANGE_COMPRESSION_MAX:
                range_compression = (range_compression[0], RANGE_COMPRESSION_MAX)
                param['range_compression']['flag'] = 2
    elif range_compression is not None: param['range_compression']['flag'] = 1
    
    # checking flip_lr
    if ((flip_lr != 'all') & 
        (flip_lr != 'random') & 
        (flip_lr is not None)): 
        param['flip_lr']['flag'] = 1
    
    # checking flip_ud
    if ((flip_ud != 'all') & 
        (flip_ud != 'random') & 
        (flip_ud is not None)): 
        param['flip_ud']['flag'] = 1
    
    # checking enhance
    if ((enhance != 'all') & 
        (enhance != 'random') & 
        (enhance is not None)): 
        param['enhance']['flag'] = 1
    
    # checking bbox_discard_thr
    if type(bbox_discard_thr) is float: 
        if ((bbox_discard_thr < BBOX_DISCARD_THR_MIN) | 
            (bbox_discard_thr > BBOX_DISCARD_THR_MAX)): 
            param['bbox_discard_thr']['flag'] = 1
    else: param['bbox_discard_thr']['flag'] = 1
    
    # checking bboxes
    if type(bboxes) is list:
        for bbox in bboxes:
            if type(bbox) is dict:
                keys = list(bbox.keys())
                if (('class_id' not in keys) | 
                    ('top' not in keys) | 
                    ('left' not in keys) | 
                    ('height' not in keys) | 
                    ('width' not in keys)):
                    param['bboxes']['flag'] = 1
    elif bboxes is not None: param['bboxes']['flag'] = 1

    # adjusting values and printing warnings
    ls_ranges = [
        'range_scale', 
        'range_translation', 
        'range_rotation', 
        'range_shear', 
        'range_noise', 
        'range_brightness', 
        'range_colorfulness', 
        'range_color_temperature',
        'range_compression'
    ]
    ls_others = [
        'flip_lr',
        'flip_ud',
        'enhance'
    ]
    
    for key, p in param.items():
        if p['flag'] > 0:
            
            # if a flag is raised, fall back to None
            if p['flag'] == 1:
                if key == 'range_scale': range_scale = None
                elif key == 'range_translation': range_translation = None
                elif key == 'range_rotation': range_rotation = None
                elif key == 'range_shear': range_shear = None
                elif key == 'range_noise': range_noise = None
                elif key == 'range_brightness': range_brightness = None
                elif key == 'range_colorfulness': range_colorfulness = None
                elif key == 'range_color_temperature': range_color_temperature = None
                elif key == 'range_compression': range_compression = None
                elif key == 'flip_lr': flip_lr = None
                elif key == 'flip_ud': flip_ud = None
                elif key == 'enhance': enhance = None
                elif key == 'bbox_discard_thr': bbox_discard_thr = BBOX_DISCARD_THR_DEFAULT
                elif key == 'bboxes': bboxes = None
            
            # handle warning messages
            if verbose is True: 
                if key in ls_ranges:
                    if p['flag'] == 1: print('WARNING!', key, 'is not a tuple of size 2. Switching to', key, '= None...')
                    elif p['flag'] == 2: print('WARNING!', key, 'is out of range! Truncating to [', p['min'], ',', p['max'], ']')
                elif key in ls_others:
                    print('WARNING!', key, 'is not "all", "random" or None. Switching to', key, '= None...')
                elif key == 'bbox_discard_thr':
                    print('WARNING!', key, 'is not a float in the interval [0,1]. Falling back to', key, '=', BBOX_DISCARD_THR_DEFAULT)
                elif key == 'bboxes':
                    print('Problem with provided bounding boxes!')
                    print('Please make sure you include a list of dictionaries with these fields: "class_id", "top", "left", "height", "width"')
                    print('Ignoring provided bounding boxes...')
                    
                
    #------------------------------------------------- 
    
    # convert bboxes to x,y coordinates
    if bboxes is not None:
        
        ls_bboxes_coord = []
        
        for i in range(len(bboxes)):
            
            x_up_left = bboxes[i]['left']
            y_up_left = bboxes[i]['top']
            x_down_right = x_up_left + bboxes[i]['width']  # x_up_left + width
            y_down_right = y_up_left + bboxes[i]['height']  # y_up_left + height
            
            ls_bboxes_coord.append([
                [x_up_left, y_up_left], 
                [x_down_right, y_up_left], 
                [x_down_right, y_down_right], 
                [x_up_left, y_down_right]
                ])
    
    #------------------------------------------------------- get random values
        
    # set random seed
    if random_seed is not None: np.random.seed(random_seed)  
    
    # degrees to radians
    if range_rotation is not None: range_rotation = np.radians(range_rotation)
    if range_shear is not None: range_shear = np.radians(range_shear)
    
    # get random values
    if range_scale is not None:
        param_scale = np.random.uniform(
            low=range_scale[0], 
            high=range_scale[1], 
            size=how_many
            )
    else:
        param_scale = np.random.uniform(
            low=1, 
            high=1, 
            size=how_many
            )
    if range_translation is not None:
        param_trans = np.random.uniform(
            low=range_translation[0], 
            high=range_translation[1], 
            size=(how_many,2)
            ).astype(int)
    else:
        param_trans = np.random.uniform(
            low=0, 
            high=0, 
            size=(how_many,2)
            ).astype(int)
    if range_rotation is not None:
        param_rot = np.random.uniform(
            low=range_rotation[0], 
            high=range_rotation[1], 
            size=how_many
            )
    else:
        param_rot = np.random.uniform(
            low=0, 
            high=0, 
            size=how_many
            )
    if range_shear is not None:
        param_shear = np.random.uniform(
            low=range_shear[0], 
            high=range_shear[1], 
            size=how_many
            )
    else:
        param_shear = np.random.uniform(
            low=0, 
            high=0, 
            size=how_many
            )
    if range_noise is not None:
        param_noise = np.random.uniform(
            low=range_noise[0], 
            high=range_noise[1], 
            size=how_many
            )
    else:
        param_noise = np.random.uniform(
            low=0, 
            high=0, 
            size=how_many
            )
    if range_brightness is not None:
        param_gain = np.random.uniform(
            low=range_brightness[0], 
            high=range_brightness[1], 
            size=how_many
            )
    else:
        param_gain = np.random.uniform(
            low=1, 
            high=1, 
            size=how_many
            )
    if range_colorfulness is not None:
        param_colorfulness = np.random.uniform(
            low=range_colorfulness[0], 
            high=range_colorfulness[1], 
            size=how_many
            )
    else:
        param_colorfulness = np.random.uniform(
            low=1, 
            high=1, 
            size=how_many
            )
    if range_color_temperature is not None:
        param_color_temperature = np.random.uniform(
            low=range_color_temperature[0], 
            high=range_color_temperature[1], 
            size=how_many
            )
    else:
        param_color_temperature = np.random.uniform(
            low=0, 
            high=0, 
            size=how_many
            )
    if range_compression is not None:
        param_compression = np.random.randint(
            low=range_compression[0], 
            high=range_compression[1], 
            size=how_many
            )
    else:
        param_compression = [None for i in range(how_many)]
    
    #-------------------------------------------- process all image variations
    
    # initiate output dcionary
    dc_augm = {}
    dc_augm['Images'] = []
    dc_augm['bboxes'] = []
    dc_augm['bboxes_discarded'] = []
    dc_augm['Transformations'] = []
    
    
    # for all images
    for i in range(how_many):
        
        dc_augm['bboxes'].append([])
        dc_augm['bboxes_discarded'].append([])
        image_transformed = img_as_float(image.copy())
    
        # configure an affine transform based on the random values
        tform = AffineTransform(
                scale=(param_scale[i],param_scale[i]),          
                rotation=param_rot[i], 
                shear=param_shear[i],
                translation=(param_trans[i,0], param_trans[i,1])
                )

        image_transformed = warp(   # warp image (pixel range -> float [0,1])
                image,       
                tform.inverse, 
                mode = 'symmetric'
                )
        
        # add color temperature variations
        if range_color_temperature is not None:
            image_transformed = convert_image_temperature(
                image_transformed, 
                degree=param_color_temperature[i]
            )
        
        # add colorfulness variations
        if range_colorfulness is not None:
            image_transformed = adjust_image_colorfulness(
                image_transformed, 
                degree=param_colorfulness[i]
            )
        
        # add gaussian noise
        if range_noise is not None:
            image_transformed = random_noise(
                image_transformed, 
                mode='gaussian', 
                seed=random_seed, 
                clip=True,
                var=param_noise[i]
            )
        
        # add brightness variations
        if range_brightness is not None:
            image_transformed = image_transformed * param_gain[i]
            
        # add compression variations
        if range_compression is not None:
            image_transformed = compress_image(
                image_transformed, 
                compression_level=param_compression[i]
            )
        
        # convert range back to [0,255]
        image_transformed = cast_image_as_uint8(image_transformed)
        
        # add transforamtions to the dictionary 
        dc_augm['Images'].append(image_transformed)
        
        dc_transf = {}
        dc_transf['Scale'] = param_scale[i]
        dc_transf['Translation'] = param_trans[i]
        dc_transf['Rotation'] = np.degrees(param_rot[i])
        dc_transf['shear'] = np.degrees(param_shear[i])
        dc_transf['Noise'] = param_noise[i]
        dc_transf['Colorfulness'] = param_colorfulness[i]
        dc_transf['Color_Temperature'] = param_color_temperature[i]
        dc_transf['JPEG_Compression'] = param_compression[i]
        dc_transf['Brightness'] = param_gain[i]
        dc_transf['Flip_lr'] = False
        dc_transf['Flip_ud'] = False
        dc_transf['Enhance'] = False
        dc_transf['Matrix'] = tform.params
        
        dc_augm['Transformations'].append(dc_transf)
          
    #------------- transform bboxes to the new coordinates of the warped image
    
        if bboxes is not None:
            
            ls_bboxes_coord_new = copy.deepcopy(ls_bboxes_coord)
            
            for b in range(len(ls_bboxes_coord_new)):
                
                for j in range(4):
                    ls_bboxes_coord_new[b][j].append(1)  # [x,y,1]
                    vector = np.array(ls_bboxes_coord_new[b][j])  
                    new_coord = np.matmul(tform.params, vector)
                    ls_bboxes_coord_new[b][j][0] = int(round(new_coord[0]))# x
                    ls_bboxes_coord_new[b][j][1] = int(round(new_coord[1]))# y

                # get the final bboxes from the new (transformed) coordinates
                # (find the min and max of the transformed xy coordinates)
                # TODO: add a diminishing factor for skewed bboxes to address
                # the fact that bboxes from highly skewed images expand!
                x_up_left = min(
                    ls_bboxes_coord_new[b][0][0], 
                    ls_bboxes_coord_new[b][1][0], 
                    ls_bboxes_coord_new[b][2][0], 
                    ls_bboxes_coord_new[b][3][0]
                    )
                y_up_left = min(
                    ls_bboxes_coord_new[b][0][1], 
                    ls_bboxes_coord_new[b][1][1], 
                    ls_bboxes_coord_new[b][2][1], 
                    ls_bboxes_coord_new[b][3][1]
                    )
                x_down_right = max(
                    ls_bboxes_coord_new[b][0][0], 
                    ls_bboxes_coord_new[b][1][0], 
                    ls_bboxes_coord_new[b][2][0], 
                    ls_bboxes_coord_new[b][3][0]
                    )
                y_down_right = max(
                    ls_bboxes_coord_new[b][0][1], 
                    ls_bboxes_coord_new[b][1][1], 
                    ls_bboxes_coord_new[b][2][1], 
                    ls_bboxes_coord_new[b][3][1]
                    )
                
    #--------------------------------------- truncate bbox to image boundaries
                
                flag_truncated = False
                
                if bbox_truncate is True:
                    
                    im_width = image_transformed.shape[1]
                    im_height = image_transformed.shape[0]
                    
                    if x_up_left < 0: 
                        x_up_left = 0
                        flag_truncated = True
                    elif x_up_left > im_width: 
                        x_up_left = im_width - 1
                        flag_truncated = True
                    
                    if x_down_right < 0: 
                        x_down_right = 0
                        flag_truncated = True
                    elif x_down_right > im_width: 
                        x_down_right = im_width - 1
                        flag_truncated = True
                    
                    if y_up_left < 0: 
                        y_up_left = 0
                        flag_truncated = True
                    elif y_up_left > im_height: 
                        y_up_left = im_height - 1
                        flag_truncated = True
                    
                    if y_down_right < 0: 
                        y_down_right = 0
                        flag_truncated = True
                    elif y_down_right > im_height: 
                        y_down_right = im_height - 1
                        flag_truncated = True

                
                width_new = x_down_right - x_up_left
                height_new = y_down_right - y_up_left
                width_old = bboxes[b]['width']
                height_old = bboxes[b]['height']
                
                # estimate how much the bbox area has changed due to cropping
                bbox_surface_ratio = ((width_new * height_new) / 
                                     (width_old * height_old * param_scale[i]))

    #------------------------------------------------------ store the new bbox
                
                if (flag_truncated == True):
                    if (bbox_surface_ratio > bbox_discard_thr):
                        dc_augm['bboxes'][i].append(
                            {
                                'left': x_up_left,
                                'top': y_up_left, 
                                'width': width_new, 
                                'height': height_new,
                                'class_id': bboxes[b]['class_id']
                            }
                        )
                    else:
                        dc_augm['bboxes_discarded'][i].append(
                            {
                                'left': x_up_left,
                                'top': y_up_left, 
                                'width': width_new, 
                                'height': height_new,
                                'class_id': bboxes[b]['class_id'],
                                'original_order': b
                            }
                        )
                else:  # keep all bboxes
                    dc_augm['bboxes'][i].append(
                        {
                            'left': x_up_left,
                            'top': y_up_left, 
                            'width': width_new, 
                            'height': height_new,
                            'class_id': bboxes[b]['class_id']
                        }
                    )
    
                    
    #-------------------------------------------------- flip images left-right
    
    if flip_lr is not None:
        
        if flip_lr == 'random':
            
            for i in range(len(dc_augm['Images'])):
            
                if random.choice([True, False]):  # random boolean
                    
                    dc_augm['Transformations'][i]['Flip_lr'] = True
            
                    dc_augm['Images'][i] = flip_image(
                        image=dc_augm['Images'][i],
                        direction='lr'
                        )
                    
                    if bboxes is not None: 
                        dc_augm['bboxes'][i] = flip_bboxes(
                            ls_bboxes=dc_augm['bboxes'][i],
                            image_width=dc_augm['Images'][i].shape[1],
                            image_height=dc_augm['Images'][i].shape[0],
                            direction='lr'
                            )
                    
        else:  # flip_lr == 'all':
            
            ls_flipped_images = []
            ls_flipped_bboxes = []
            dc_augm['Transformations'].extend(
                copy.deepcopy(
                    dc_augm['Transformations']
                    )
                )
            
            for i in range(len(dc_augm['Images'])):
                    
                dc_augm['Transformations'][i + how_many]['Flip_lr'] = True
        
                ls_flipped_images.append(
                    flip_image(
                        image=dc_augm['Images'][i],
                        direction='lr'
                        )
                    )
                
                ls_flipped_bboxes.append(
                    flip_bboxes(
                        ls_bboxes=dc_augm['bboxes'][i],
                        image_width=dc_augm['Images'][i].shape[1],
                        image_height=dc_augm['Images'][i].shape[0],
                        direction='lr'
                        )
                    )
            
            dc_augm['Images'].extend(ls_flipped_images)
            dc_augm['bboxes'].extend(ls_flipped_bboxes)

    #----------------------------------------------------- flip images up-down
    
    if flip_ud is not None:
        
        if flip_ud == 'random':
            
            for i in range(len(dc_augm['Images'])):
            
                if random.choice([True, False]):  # random boolean
                    
                    dc_augm['Transformations'][i]['Flip_ud'] = True
            
                    dc_augm['Images'][i] = flip_image(
                        image=dc_augm['Images'][i],
                        direction='ud'
                        )
                    
                    if bboxes is not None: 
                        dc_augm['bboxes'][i] = flip_bboxes(
                            ls_bboxes=dc_augm['bboxes'][i],
                            image_width=dc_augm['Images'][i].shape[1],
                            image_height=dc_augm['Images'][i].shape[0],
                            direction='ud'
                            )
                    
        else:  # flip_ud == 'all':
            
            ls_flipped_images = []
            ls_flipped_bboxes = []
            dc_augm['Transformations'].extend(
                copy.deepcopy(
                    dc_augm['Transformations']
                    )
                )
            
            for i in range(len(dc_augm['Images'])):
                    
                dc_augm['Transformations'][i + how_many]['Flip_ud'] = True
        
                ls_flipped_images.append(
                    flip_image(
                        image=dc_augm['Images'][i],
                        direction='ud'
                        )
                    )
                
                ls_flipped_bboxes.append(
                    flip_bboxes(
                        ls_bboxes=dc_augm['bboxes'][i],
                        image_width=dc_augm['Images'][i].shape[1],
                        image_height=dc_augm['Images'][i].shape[0],
                        direction='ud'
                        )
                    )
            
            dc_augm['Images'].extend(ls_flipped_images)
            dc_augm['bboxes'].extend(ls_flipped_bboxes)
                    
                
    #---------------------------------------------------------- enhance images
    
    if enhance is not None:
        
        if enhance == 'random':
            
            for i in range(len(dc_augm['Images'])):
            
                if random.choice([True, False]):  # random boolean
                    
                    dc_augm['Transformations'][i]['Enhance'] = True
            
                    dc_augm['Images'][i] = equalize_adapthist(
                        dc_augm['Images'][i], 
                        kernel_size=None, 
                        clip_limit=0.01, 
                        nbins=256
                    )                    
                    
        else:  # enhance == 'all':
            
            ls_enhanced_images = []
            ls_enhanced_bboxes = []
            dc_augm['Transformations'].extend(
                copy.deepcopy(
                    dc_augm['Transformations']
                )
            )
            
            for i in range(len(dc_augm['Images'])):
                    
                dc_augm['Transformations'][i + how_many]['Enhance'] = True
        
                ls_enhanced_images.append(
                    equalize_adapthist(
                        dc_augm['Images'][i], 
                        kernel_size=None, 
                        clip_limit=0.01, 
                        nbins=256
                    )
                )
                
                ls_enhanced_bboxes.append(dc_augm['bboxes'][i])
            
            dc_augm['Images'].extend(ls_enhanced_images)
            dc_augm['bboxes'].extend(ls_enhanced_bboxes)
            
    #------------------------------------------------- visualize augmentations

    if display is not False:
        
        total_figs = len(dc_augm['Images'])
        if display is True: display = 'single'
        if ((display == 'grid') & (total_figs > 10)):
            display == 'single'  # cannot display in grid more than 10 images
        
        if (display == 'single') | (display == 'grid'):
            
            # show original image
            print('\nAugmenting', image_filename.split('/')[-1], end='')
            print(' [x', end='')
            print(how_many, end='')
            if flip_lr=='all': print(' x2', end='')
            if flip_ud=='all': print(' x2', end='')
            if enhance=='all': print(' x2', end='')
            print(']')
            visualize_image(
                image, 
                title='Original', 
                bboxes=bboxes, 
                max_number_of_classes=max_number_of_classes
            )
            
            if display == 'grid':
                depict = False
                SIZE_FONT = 8
                WIDTH_LINE = 2
                DPI = 100
                BLOCK_SIZE = 5
                HEIGHT = 17
                fig = plt.figure(figsize=(total_figs * BLOCK_SIZE, HEIGHT), dpi=DPI)
            else:
                depict = True
        
            # show augmentations
            for i,image_transformed in enumerate(dc_augm['Images']):
                
                if display == 'grid':
                    plt.subplot(1, total_figs, i+1)
                    title = ''
                    title += '\nAugmentation ' + str(i+1)
                    title += '\nScale:' + str(dc_augm['Transformations'][i]['Scale'])
                    title += '\nTranslation_x:' + str(dc_augm['Transformations'][i]['Translation'][0])
                    title += '\nTranslation_y:' + str(dc_augm['Transformations'][i]['Translation'][1])
                    title += '\nRotation:' + str(dc_augm['Transformations'][i]['Rotation'])
                    title += '\nshear:' + str(dc_augm['Transformations'][i]['shear'])
                    title += '\nNoise:' + str(dc_augm['Transformations'][i]['Noise'])
                    title += '\nBrightness:' + str(dc_augm['Transformations'][i]['Brightness'])
                    title += '\nColorfulness:' + str(dc_augm['Transformations'][i]['Colorfulness'])
                    title += '\nColor Temperature:' + str(dc_augm['Transformations'][i]['Color_Temperature'])
                    title += '\nJPEG Compression:' + str(dc_augm['Transformations'][i]['JPEG_Compression'])
                    title += '\nEnhance:' + str(dc_augm['Transformations'][i]['Enhance'])
                    title += '\nFlip left->right:' + str(dc_augm['Transformations'][i]['Flip_lr'])
                    title += '\nFlip up->down:' + str(dc_augm['Transformations'][i]['Flip_ud'])
                else:
                    title = 'Augmentation ' + str(i+1)
                    print('\nTransformation for augmentation', i+1)
                    print('Scale:', 
                          dc_augm['Transformations'][i]['Scale'])
                    print('Translation_x:', 
                          dc_augm['Transformations'][i]['Translation'][0])
                    print('Translation_y:', 
                          dc_augm['Transformations'][i]['Translation'][1])
                    print('Rotation:', 
                          dc_augm['Transformations'][i]['Rotation'])
                    print('shear:', 
                          dc_augm['Transformations'][i]['shear'])
                    print('Noise:', 
                          dc_augm['Transformations'][i]['Noise'])
                    print('Brightness:', 
                          dc_augm['Transformations'][i]['Brightness'])
                    print('Colorfulness:', 
                          dc_augm['Transformations'][i]['Colorfulness'])
                    print('Color Temperature:', 
                          dc_augm['Transformations'][i]['Color_Temperature'])
                    print('JPEG Compression:', 
                          dc_augm['Transformations'][i]['JPEG_Compression'])
                    print('Enhance:', 
                          dc_augm['Transformations'][i]['Enhance'])
                    print('Flip left->right: ', 
                          dc_augm['Transformations'][i]['Flip_lr'])
                    print('Flip up->down: ', 
                          dc_augm['Transformations'][i]['Flip_ud'])
                

                visualize_image(
                    image_transformed, 
                    title=title, 
                    bboxes=dc_augm['bboxes'][i], 
                    max_number_of_classes=max_number_of_classes,
                    display = depict
                )
            plt.show()
    
    if bboxes is None: 
        del dc_augm['bboxes']
        del dc_augm['bboxes_discarded']
    
    return dc_augm








def load_file(file_uri, file_type):
    """
    ---------------------------------------------------------------------------
           Loads files either from S3 or locally (image or manifest txt)
    ---------------------------------------------------------------------------
    
    INPUTS
    ------
    file_uri: string 
        Location of the file to be loaded. This could be either an S3 URI or 
        a path to a local file. 
    file_type: string
        The type of file to be loaded. Can either be 'image' or 'manifest'
        txt file. 
    """
    
    if file_uri[:5] == 's3://':  # if file is in S3
        
        # parse uri to find bucket and key
        slash = file_uri[5:].find('/')
        bucket = file_uri[5 : 5+slash]
        key = file_uri[5+slash+1:]
        
        # read the raw bytes of the file
        s3 = boto3.client('s3')      
        raw_data = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        
        if file_type == 'manifest':
            raw_data = raw_data.decode('utf-8')
            # convert raw string to json lines (list of strings)
            file_content = []
            new_line = raw_data.find('\n')
            while new_line != -1:
                file_content.append(raw_data[:new_line+1])
                raw_data = raw_data[new_line+1:]
                new_line = raw_data.find('\n')
                
        elif file_type == 'image':
            q = Image.open(io.BytesIO(raw_data))
            file_content = np.array(q)
            
        else:
            print('Problem! Unknown file type!') 
            print('Parameter file_type can either be "manifest" or "image"!')

    else:  # if file is local
        
        if file_type == 'manifest':
            with open(file_uri) as f:  # open the manifest file
                file_content = f.readlines()
        elif file_type == 'image':
            file_content = imageio.imread(file_uri)
        else:
            print('Problem! Unknown file type!') 
            print('Parameter file_type can either be "manifest" or "image"!')
       
    return file_content






def save_file(file_data, file_uri, file_type):
    """
    ---------------------------------------------------------------------------
            Saves files either to S3 or locally (image or manifest txt)
    ---------------------------------------------------------------------------
    
    INPUTS
    ------
    file_data: string or numpy array (refering to pixels) 
        The actual file data that will be saved, either locally or in S3. 
    file_uri: string 
        Location where the file will be saved. This could be either an S3 URI 
        or a local path to a local file. 
    file_type: string
        The type of file to be saved. Can either be 'image' or 'manifest'
        txt file. 
    """
    JPG_QUALITY = 95  # [1,100]  higher -> better quality and larger size
    
    if file_uri[:5] == 's3://':  # if file will be saved in S3
        
        # parse uri to find bucket and key
        slash = file_uri[5:].find('/')
        bucket = file_uri[5 : 5+slash]
        key = file_uri[5+slash+1:]
        
        # get the s3 client object
        s3 = boto3.client('s3')      
        
        # if file is a manifest
        if file_type == 'manifest':
            # Convert the list of JSON objects to a string with one object per line
            json_lines_str = "\n".join([line for line in file_data])
            # Create a BytesIO object containing the string
            json_lines_bytes = io.BytesIO(json_lines_str.encode())
            s3.put_object(
                Body=json_lines_bytes, 
                Bucket=bucket, 
                Key=key
            )
        
        # if file is an image
        elif file_type == 'image':
            file_stream = io.BytesIO()
            image = Image.fromarray(file_data)
            # if image is PNG
            if (file_uri[-4:] == '.png') | (file_uri[-4:] == '.PNG'):
                image.save(file_stream, format='png') 
            else:  # assuming JPG
                image.save(file_stream, format='jpeg', quality=JPG_QUALITY)
            s3.put_object(
                Body=file_stream.getvalue(), 
                Bucket=bucket, 
                Key=key
            )
         
        else:
            print('Problem! Unknown file type!') 
            print('Parameter file_type can either be "manifest" or "image"!')

    else:  # if file will be saved locally
        
        if file_type == 'manifest':
            with open(file_uri, "w") as f:
                for line in file_data:
                    f.write(f"{line}\n")  
            
        elif file_type == 'image':
            # if image is PNG
            if (file_uri[-4:] == '.png') | (file_uri[-4:] == '.PNG'):
                imageio.imsave(
                    file_uri, 
                    file_data
                )
            else:  # assuming JPG
                imageio.imsave(
                    file_uri, 
                    file_data, 
                    quality=JPG_QUALITY
                )
        else:
            print('Problem! Unknown file type!') 
            print('Parameter file_type can either be "manifest" or "image"!')




def break_filename(filename):
    # breaks up a filename to its components
    file = {}
    filename_object = Path(filename)  # path + name + extension
    file['filename_no_path'] = str(filename_object.name)  # name + extension (no path)
    file['extension'] = str(filename_object.suffix)  # extension (no path, no name)
    file['filename_no_extension'] = str(filename_object.stem)  # path + name (no extension)
    file['filename_only'] = str(Path(file['filename_no_path']).stem)  # name (no path, no extension)
    return file
    
    

def fix_uri_double_shashes(file_uri):
    # replaces any double slashes with single ones in an uri
    if file_uri[:5] == 's3://':  # if file is in S3
        start = 5
        fixed_uri = file_uri[:5] + file_uri[5:].replace('//', '/')
    else:
        fixed_uri = file_uri.replace('//', '/')
    return fixed_uri



def augment_dataset( 
    uri_manifest_file,
    uri_destination,
    ls_class_names,
    filename_postfix = '_augm_',
    include_original=True,
    verbose = True,
    **augm_param
    ):
    
    """
    ---------------------------------------------------------------------------
               Augments a whole dataset based on its manifest file
    ---------------------------------------------------------------------------
    The function supports manifest files compatible/generated by Amazon 
    SageMaker Ground Truth for 3 use-cases: 
        A) object detection
        B) single label image classification
        C) multi label image classification
    Supported image types are JPG/JPEG and PNG.
    
    INPUTS
    ------
    uri_manifest_file: string
        The URI where the manifest file of the dataset resides. It can either 
        be a local relative path e.g. folder1/folder2/dataset.manifest or a
        S3 one, e.g. s3://bucket-name/folder3/datsaset.manifest
    uri_destination: string 
        The location where the augmented dataset (images + new manifest file)
        will be saved. Similar to uri_manifest_file, it can be either a local 
        relative path e.g. folder1/folder2 or a S3 one, e.g. 
        s3://bucket-name/folder3
    ls_class_names: list of strings
        The names of the classes (objects or image classes) that are possible
        to be included in the dataset.
    filename_postfix: string
        The post-fix that will be added for all the new augmented images and
        the new manifest file of the augmented dataset.
    include_original: boolean
        Whether or not to include the original images in the augmented 
        dataset. 
    verbose: boolean
        Whether or not to print statements showing the progress of the 
        augmentation process.
    **augm_param: dictionary
        A dictionary of all the parameters that the augment_image()function
        requires in order to augment the original dataset. These parameters
        describe the number and type of augmentations that will be applied. 
    
    OUTPUTS
    ------
    stats: dictionary
        A dictionary of summary statistics regarding the original and the 
        augmented dataset. It includes the number of images, as well as the 
        number of examples per object (for object detection). 
    
    """

    # initializations
    new_manifest = []  # the new manifest file for the augmented dataset
    n_samples_augmented = 0
    n_samples_original = 0
    class_histogram_original = np.zeros(len(ls_class_names), dtype=int)
    class_histogram_augmented = np.zeros(len(ls_class_names), dtype=int)
    
    lines = load_file(
        file_uri=uri_manifest_file, 
        file_type='manifest'
    )
      
    # process json lines (corresponding to one image) one by one
    for l,line in enumerate(lines):
        line_dict = json.loads(line)  # load one json line
        file = break_filename(line_dict['source-ref'])  # filename of each input image
        ls_keys = list(line_dict.keys())

        if verbose is True:
            print('Augmenting image', l+1, 'out of', len(lines), '[', round(((l+1)*100)/(len(lines)),2), '%]')
            # print(file['filename_no_path'])

        # understand dictionary keys
        # assumption: source is fixed and the metadata is always the annotations with a '-metadata' sufix
        keys_source = 'source-ref'  # this one is always fixed
        keys_metadata = [key for key in ls_keys if 'metadata' in key][0]  # find the one that has metadata in its name
        keys_annotations = keys_metadata[:keys_metadata.find('-metadata')]  # find the one that has the annotations
        
        # understand what type of problem we are dealing with: 
        # single-label classification / multi-label classification / object detection
        problem_type = None
        bboxes = None
        if type(line_dict[keys_annotations]) is dict:  # check if object-detection
            if 'annotations' in list(line_dict[keys_annotations].keys()):
                # check if 'annotations' is a list of dictionaries
                if type(line_dict[keys_annotations]['annotations']) is list:  
                    bboxes = line_dict[keys_annotations]['annotations']
                    problem_type = 'object_detection'
                    for bbox in bboxes:  # check if each bbox is a dictionary
                        if type(bbox) is not dict:
                            bboxes = None
                            break
        elif type(line_dict[keys_annotations]) is list:  # check if multi-label
            problem_type = 'multi_label'
        elif type(line_dict[keys_annotations]) is int:  # check if single-label
            problem_type = 'single_label'  
        else:
            print('Problem! Unknown type of annotations in the manifest file...')
        
        # statistics about the original images
        n_samples_original += 1
        if problem_type == 'object_detection':
            for bbox in bboxes:
                class_histogram_original[int(bbox['class_id'])] += 1  # counting objects
                if include_original is True: class_histogram_augmented[int(bbox['class_id'])] += 1
        elif problem_type == 'multi_label':
            for class_id in line_dict[keys_annotations]:
                class_histogram_original[int(class_id)] += 1  # counting classes
                if include_original is True: class_histogram_augmented[int(class_id)] += 1
        elif problem_type == 'single_label':
            class_id = line_dict[keys_annotations]
            class_histogram_original[int(class_id)] += 1  # counting classes
            if include_original is True: class_histogram_augmented[int(class_id)] += 1

        # load image from source
        image_original = load_file(
            file_uri=line_dict[keys_source], 
            file_type='image'
        )
        
        # add the original image to the augmented dataset
        if include_original is True:
            n_samples_augmented += 1
            new_manifest.append(json.dumps(line_dict))
            uri_image_orig = f'{uri_destination}/{file["filename_no_path"]}'
            uri_image_orig = fix_uri_double_shashes(uri_image_orig)
            save_file(
                file_data=image_original, 
                file_uri=uri_image_orig, 
                file_type='image'
            )
            
        # resolve augmentation parameters
        ls_keys = list(augm_param.keys())
        if 'max_number_of_classes' in ls_keys: augm_max_number_of_classes = augm_param['max_number_of_classes'][0]
        else: augm_max_number_of_classes = None
        if 'how_many' in ls_keys: augm_how_many = augm_param['how_many'][0]
        else: augm_how_many = 1
        if 'random_seed' in ls_keys: augm_random_seed = augm_param['random_seed'][0]
        else: augm_random_seed = None
        if 'range_scale' in ls_keys: augm_range_scale = augm_param['range_scale'][0]
        else: augm_range_scale = None
        if 'range_translation' in ls_keys: augm_range_translation = augm_param['range_translation'][0]
        else: augm_range_translation = None
        if 'range_rotation' in ls_keys: augm_range_rotation = augm_param['range_rotation'][0]
        else: augm_range_rotation = None
        if 'range_shear' in ls_keys: augm_range_shear = augm_param['range_shear'][0]
        else: augm_range_shear = None
        if 'range_noise' in ls_keys: augm_range_noise = augm_param['range_noise'][0]
        else: augm_range_noise = None
        if 'range_brightness' in ls_keys: augm_range_brightness = augm_param['range_brightness'][0]
        else: augm_range_brightness = None
        if 'range_colorfulness' in ls_keys: augm_range_colorfulness = augm_param['range_colorfulness'][0]
        else: augm_range_colorfulness = None
        if 'range_color_temperature' in ls_keys: augm_range_color_temperature = augm_param['range_color_temperature'][0]
        else: augm_range_color_temperature = None
        if 'range_compression' in ls_keys: augm_range_compression = augm_param['range_compression'][0]
        else: augm_range_compression = None
        if 'flip_lr' in ls_keys: augm_flip_lr = augm_param['flip_lr'][0]
        else: augm_flip_lr = None
        if 'flip_ud' in ls_keys: augm_flip_ud = augm_param['flip_ud'][0]
        else: augm_flip_ud = None
        if 'enhance' in ls_keys: augm_enhance = augm_param['enhance'][0]
        else: augm_enhance = None
        if 'bbox_truncate' in ls_keys: augm_bbox_truncate = augm_param['bbox_truncate'][0]
        else: augm_bbox_truncate = True
        if 'bbox_discard_thr' in ls_keys: augm_bbox_discard_thr = augm_param['bbox_discard_thr'][0]
        else: augm_bbox_discard_thr = 0.85
        
        # generate augmented images
        augmentations = augment_image(
            image_original,
            bboxes = bboxes,
            max_number_of_classes=augm_max_number_of_classes,
            how_many=augm_how_many,
            random_seed=augm_random_seed,
            range_scale=augm_range_scale, 
            range_translation=augm_range_translation,
            range_rotation=augm_range_rotation,
            range_shear=augm_range_shear,
            range_noise=augm_range_noise,
            range_brightness=augm_range_brightness,
            range_colorfulness=augm_range_colorfulness,
            range_color_temperature=augm_range_color_temperature,
            range_compression=augm_range_compression,
            flip_lr = augm_flip_lr,
            flip_ud = augm_flip_ud,
            enhance = augm_enhance,
            bbox_truncate = augm_bbox_truncate,
            bbox_discard_thr = augm_bbox_discard_thr,
            display=False,
            verbose=False
        )

        # go through all the generated images
        for i,image in enumerate(augmentations['Images']):
            
            # new image size of augmented image
            if problem_type == 'object_detection':
                image_height = image.shape[0]
                image_width = image.shape[1]
                if len(image.shape) == 3:
                    image_depth = image.shape[2]
                else:
                    image_depth = 1
                line_dict[keys_annotations]['image_size'] = [
                    {
                    "width": image_width, 
                    "height": image_height, 
                    "depth": image_depth
                    }
                ]

            # reconstruct filename of augmented image
            uri_image_augm = f'{uri_destination}/{file["filename_only"]}{filename_postfix}{str(i+1)}{file["extension"]}'
            uri_image_augm = fix_uri_double_shashes(uri_image_augm)
            
            # save augmented image
            save_file(
                file_data=image, 
                file_uri=uri_image_augm, 
                file_type='image'
            )
            
            # add new filename to the manifest file
            line_dict[keys_source] = uri_image_augm

            # update bounding boxes and metadata for the augmented images
            if problem_type == 'object_detection':
                
                # update bounding boxes
                line_dict[keys_annotations]['annotations'] = augmentations['bboxes'][i]

                # update metadata objects
                # a list of all the original indices of the discarded objects
                ls_discarded_indices = [ 
                    discarded_bbox['original_order'] 
                    for discarded_bbox 
                    in augmentations['bboxes_discarded'][i] 
                ]
                # throw away any objects coming from discarded objects
                ls_new_objects = [ 
                    conf 
                    for indx,conf 
                    in enumerate(line_dict[keys_metadata]['objects']) 
                    if indx not in ls_discarded_indices 
                ]
                line_dict[keys_metadata]['objects'] = ls_new_objects  # update

                # update class map
                ls_classes = [bbox['class_id'] for bbox in augmentations['bboxes'][i]]
                unique_classes = set(ls_classes)
                dict_new_class_map = { str(cl): ls_class_names[cl] for cl in unique_classes}
                line_dict[keys_metadata]['class-map'] = dict_new_class_map

            # tracking augmentation statistics 
            n_samples_augmented += 1  # count augmetned images
            if problem_type == 'object_detection':
                for bbox in bboxes:
                    class_histogram_augmented[int(bbox['class_id'])] += 1  # counting object annotations
            elif problem_type == 'multi_label':
                for class_id in line_dict[keys_annotations]:
                    class_histogram_augmented[int(class_id)] += 1  # counting image annotations
            elif problem_type == 'single_label':
                class_id == line_dict[keys_annotations]
                class_histogram_augmented[int(class_id)] += 1  # counting image annotations
            
            # add a new json line for this augmentation image
            new_manifest.append(json.dumps(line_dict))

    # reconstruct the filename of augmented manifest file
    file = break_filename(uri_manifest_file)
    uri_manifest_augm = f'{uri_destination}/{file["filename_only"]}{filename_postfix}{file["extension"]}'
    uri_manifest_augm = fix_uri_double_shashes(uri_manifest_augm)

    # save the augmented manifest file 
    save_file(
        file_data=new_manifest, 
        file_uri=uri_manifest_augm, 
        file_type='manifest'
    )
    
    # package statistics
    stats = {}
    stats['original'] = {'n_samples': n_samples_original, 'class_hist': class_histogram_original}
    stats['augmentations'] = {'n_samples': n_samples_augmented, 'class_hist': class_histogram_augmented}

    return stats