#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Function for generating geometric affine augmentations in images, while also
handling bounding boxes.

@author: vasileios vonikakis
"""

import imageio
import copy
import random
import numpy as np
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
    # adjusts the colorfullness of an image according to the degree
    
    image = img_as_float(image)  # [0,1]
    image_gray = rgb2gray(image)  # [0,1]
    image_gray = np.dstack((image_gray, image_gray, image_gray))
    
    image_color_delta = image - image_gray  # deviations from gray
    
    image_adjusted = image_gray
    image_adjusted = image_adjusted + image_color_delta * degree
    
    return cast_image_as_uint8(image_adjusted)



def cast_image_as_uint8(image):
    # converts a float image of [0,1] to an uint8 image of [0,255]
    image *= 255
    image[image>255] = 255
    image[image<0] = 0
    return image.astype(np.uint8)
    

    
    


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
    
    return cast_image_as_uint8(image)




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






def augment_image(
        image_filename,
        bboxes = None,
        how_many=1,
        random_seed=None,
        range_scale=None, 
        range_translation=None,
        range_rotation=None,
        range_sheer=None,
        range_noise=None,
        range_brightness=None,
        range_colorfullness=None,
        range_color_temperature=None,
        flip_lr = None,
        flip_ud = None,
        enhance = None,
        bbox_truncate = True,
        bbox_discard_thr = 0.75,
        display=False
        ):
    
    '''
    ---------------------------------------------------------------------------
      Functiont that generates random affine augmentations for a given image
    ---------------------------------------------------------------------------
    The function apply affine distortions on a given image and generates many
    random variations of it. If bounding boxes are provided, then they are also
    transformed to the new distorted image and returned back. The function is
    based on the scikit-image library.
    
    INPUTS
    ------
    image_filename: string
        Filename of the input image.
    bboxes: list of dictionaries or None
        The list of the given bounding boxes in the image. A bounding box is 
        defined by 5 numbers of a dictionary: {"class_id", "top", "left", 
        "height", "width"}. For example, a possbile bounding box could be:
        {"class_id": 0, "top": 44, "left": 394, "height": 189, "width": 147}
        Top, left, height and width are defined in pixels. 
    how_many: int
        How many augmentations to generate per input image.
    random_seed: int
        Number for setting the random number generator, in order to control 
        reproducibility.
    range_scale: (min, max) tuple of float, or None
        Minimum and maximum range of possible zooming/unzooming scale factors. 
        Value range: float(0,inf), <1=zoom in, >1=zoom out, e.g. (0.5,1.5). 
        If None, then this transformation is deactivated. 
    range_translation: (min, max) tuple of integer, or None
        Minimum and maximum range for offseting the (x,y) position of the image 
        (in pixels). Value range: int[0,inf), e.g. (-100, 100). 
        If None, then this transformation is deactivated. 
    range_rotation: (min, max) tuple of float, or None
        Minimum and maximum range for rotating image left/right (in degrees). 
        Value range: float[-360,360], e.g.(-45, 45).
        If None, then this transformation is deactivated.
    range_sheer: (min, max) tuple of integer, or None
        Minimum and maximum range for skewing image left/right (in degrees). 
        Value range: float[-360,360], e.g.(-45, 45).
        If None, then this transformation is deactivated.
    range_noise: (min, max) tuple of float, or None
        Minimum and maximum range of noise variance. 
        Value range: float[0,inf), e.g. (0, 0.001). 
        If None, then this transformation is deactivated. 
    range_brightness: (min, max) tuple of float, or None
        Minimum and maximum range for brightness gain. 
        Value range: float(0,inf), 1=no change, <1=darken, >1=brighten, 
        e.g. (0.5, 1.5). If None, then this transformation is deactivated. 
    range_colorfulness: (min, max) tuple of float, or None
        Minimum and maximum range for color saturation. 
        Value range: float[0,inf), 1=no change, 0=grayscale, >1=more saturated, 
        e.g. (0.5, 1.5). If None, then this transformation is deactivated. 
    range_color_temperature: (min, max) tuple of float, or None
        Minimum and maximum range for color temperature (cool/warm). 
        Value range: float[-1,1], -1=cool, 0=no change, 1=warm, 
        e.g. (-0.5, 1.0). If None, then this transformation is deactivated. 
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
    display: boolean
        Show visualizations and details or not.
    
    OUTPUT
    ------
    dcionary containing:
        Augmented images.
        Transformed bounding boxes.
        Details about each augmentation. 
        List of discarded bounding boxes (being outside the image).
        
    '''
    
    
    #------------------------------------------------- sanity check for ranges
    
    # value ranges
    RANGE_SCALE_MIN = 0.1
    RANGE_SCALE_MAX = 5
    RANGE_ROTATION_MIN = -360
    RANGE_ROTATION_MAX = 360
    RANGE_SHEER_MIN = -360
    RANGE_SHEER_MAX = 360
    RANGE_NOISE_MIN = 0.0
    RANGE_NOISE_MAX = 3
    RANGE_BRIGHTNESS_MIN = 0.1
    RANGE_BRIGHTNESS_MAX = 10
    RANGE_COLORFULLNESS_MIN = 0.0
    RANGE_COLORFULLNESS_MAX = 5
    RANGE_COLOR_TEMPERATURE_MIN = -1
    RANGE_COLOR_TEMPERATURE_MAX = 1
    BBOX_DISCARD_THR_MIN = 0
    BBOX_DISCARD_THR_MAX = 1
    BBOX_DISCARD_THR_DEFAULT = 0.75

    if type(range_scale) is tuple:
        if len(range_scale) != 2:
            range_scale = None
        else:
            if range_scale[1] <= range_scale[0]:
                range_scale = (range_scale[1], range_scale[0])
            if range_scale[0] <= RANGE_SCALE_MIN:
                range_scale = (RANGE_SCALE_MIN, range_scale[1])
            if range_scale[1] >= RANGE_SCALE_MAX:
                range_scale = (range_scale[0], RANGE_SCALE_MAX)
    else:
        range_scale = None
                    
    if type(range_translation) is tuple:
        if len(range_translation) != 2:
            range_translation = None
        else:
            if range_translation[1] <= range_translation[0]:
                range_translation = (range_translation[1], range_translation[0])
    else:
        range_translation = None
                 
    if type(range_rotation) is tuple:
        if len(range_rotation) != 2:
            range_rotation = None
        else:
            if range_rotation[1] <= range_rotation[0]:
                range_rotation = (range_rotation[1], range_rotation[0])
            if range_rotation[0] <= RANGE_ROTATION_MIN:
                range_rotation = (RANGE_ROTATION_MIN, range_rotation[1])
            if range_rotation[1] >= RANGE_ROTATION_MAX:
                range_rotation = (range_rotation[0], RANGE_ROTATION_MAX)
    else:
        range_rotation = None
    
    if type(range_sheer) is tuple:
        if len(range_sheer) != 2:
            range_sheer = None
        else:
            if range_sheer[1] <= range_sheer[0]:
                range_sheer = (range_sheer[1], range_sheer[0])
            if range_sheer[0] <= RANGE_SHEER_MIN:
                range_sheer = (RANGE_SHEER_MIN, range_sheer[1])
            if range_sheer[1] >= RANGE_SHEER_MAX:
                range_sheer = (range_sheer[0], RANGE_SHEER_MAX)
    else:
        range_sheer = None
    
    if type(range_noise) is tuple:
        if len(range_noise) != 2:
            range_noise = None
        else:
            if range_noise[1] <= range_noise[0]:
                range_noise = (range_noise[1], range_noise[0])
            if range_noise[0] <= RANGE_NOISE_MIN:
                range_noise = (RANGE_NOISE_MIN, range_noise[1])
            if range_noise[1] >= RANGE_NOISE_MAX:
                range_noise = (range_noise[0], RANGE_NOISE_MAX)
    else:
        range_noise = None
          
    if type(range_brightness) is tuple:
        if len(range_brightness) != 2:
            range_brightness = None
        else:
            if range_brightness[1] <= range_brightness[0]:
                range_brightness = (range_brightness[1], range_brightness[0])
            if range_brightness[0] <= RANGE_BRIGHTNESS_MIN:
                range_brightness = (RANGE_BRIGHTNESS_MIN, range_brightness[1])
            if range_brightness[1] >= RANGE_BRIGHTNESS_MAX:
                range_brightness = (range_brightness[0], RANGE_BRIGHTNESS_MAX)
    else:
        range_brightness = None
        
    if type(range_colorfullness) is tuple:
        if len(range_colorfullness) != 2:
            range_colorfullness = None
        else:
            if range_colorfullness[1] <= range_colorfullness[0]:
                range_colorfullness = (range_colorfullness[1], range_colorfullness[0])
            if range_colorfullness[0] <= RANGE_COLORFULLNESS_MIN:
                range_colorfullness = (RANGE_COLORFULLNESS_MIN, range_colorfullness[1])
            if range_colorfullness[1] >= RANGE_COLORFULLNESS_MAX:
                range_colorfullness = (range_colorfullness[0], RANGE_COLORFULLNESS_MAX)
    else:
        range_colorfullness = None
    
    if type(range_color_temperature) is tuple:
        if len(range_color_temperature) != 2:
            range_color_temperature = None
        else:
            if range_color_temperature[1] <= range_color_temperature[0]:
                range_color_temperature = (range_color_temperature[1], range_color_temperature[0])
            if range_color_temperature[0] <= RANGE_COLOR_TEMPERATURE_MIN:
                range_color_temperature = (RANGE_COLOR_TEMPERATURE_MIN, range_color_temperature[1])
            if range_color_temperature[1] >= RANGE_COLOR_TEMPERATURE_MAX:
                range_color_temperature = (range_color_temperature[0], RANGE_COLOR_TEMPERATURE_MAX)
    else:
        range_color_temperature = None
          
    if type(flip_lr) is str: 
        if (flip_lr != 'all') & (flip_lr != 'random'):
            flip_lr = None
            
    if type(flip_ud) is str: 
        if (flip_ud != 'all') & (flip_ud != 'random'):
            flip_ud = None
            
    if type(enhance) is str: 
        if (enhance != 'all') & (enhance != 'random'):
            enhance = None
            
    if type(bbox_discard_thr) is float: 
        if bbox_discard_thr<BBOX_DISCARD_THR_MIN: 
            bbox_discard_thr = BBOX_DISCARD_THR_MIN
        if bbox_discard_thr>BBOX_DISCARD_THR_MAX: 
            bbox_discard_thr = BBOX_DISCARD_THR_MAX
    else:
        bbox_discard_thr = BBOX_DISCARD_THR_DEFAULT
    

    if type(bboxes) is list:
        for bbox in bboxes:
            if type(bbox) is dict:
                keys = list(bbox.keys())
                if (('class_id' not in keys) | 
                    ('top' not in keys) | 
                    ('left' not in keys) | 
                    ('height' not in keys) | 
                    ('width' not in keys)):
                    
                    print('Problem with provided bounding boxes!')
                    print('Please make sure you include a list of dictionaries with these fields: "class_id", "top", "left", "height", "width"')
                    print('Ignoring provided bounding boxes...')
                    bboxes = None          
    else:
        bboxes = None

    
    #------------------------------------------------- 
    
    # load image
    image = imageio.imread(image_filename)
    
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
    if range_sheer is not None: range_sheer = np.radians(range_sheer)
    
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
    if range_sheer is not None:
        param_sheer = np.random.uniform(
            low=range_sheer[0], 
            high=range_sheer[1], 
            size=how_many
            )
    else:
        param_sheer = np.random.uniform(
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
    if range_colorfullness is not None:
        param_colorfullness = np.random.uniform(
            low=range_colorfullness[0], 
            high=range_colorfullness[1], 
            size=how_many
            )
    else:
        param_colorfullness = np.random.uniform(
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
    
        # configure an affine transform based on the random values
        tform = AffineTransform(
                scale=(param_scale[i],param_scale[i]),          
                rotation=param_rot[i], 
                shear=param_sheer[i],
                translation=(param_trans[i,0], param_trans[i,1])
                )
        
        image_transformed = warp(   # warp image (pixel range -> float [0,1])
                image,       
                tform.inverse, 
                mode = 'symmetric'
                )
        
        # add color temperature variations
        image_transformed = convert_image_temperature(
            image_transformed, 
            degree=param_color_temperature[i]
        )
        
        # add colorfullness variations
        image_transformed = adjust_image_colorfulness(
            image_transformed, 
            degree=param_colorfullness[i]
        )
        
        # add gaussian noise
        image_transformed = random_noise(
            image_transformed, 
            mode='gaussian', 
            seed=random_seed, 
            clip=True,
            var=param_noise[i]
        )
        
        # add brightness variations
        image_transformed = image_transformed * param_gain[i]
        
        # convert range back to [0,255]
        image_transformed = cast_image_as_uint8(image_transformed)
        
        # add transforamtions to the dictionary 
        dc_augm['Images'].append(image_transformed)
        
        dc_transf = {}
        dc_transf['Scale'] = param_scale[i]
        dc_transf['Translation'] = param_trans[i]
        dc_transf['Rotation'] = np.degrees(param_rot[i])
        dc_transf['Sheer'] = np.degrees(param_sheer[i])
        dc_transf['Noise'] = param_noise[i]
        dc_transf['Colorfullness'] = param_colorfullness[i]
        dc_transf['Color_Temperature'] = param_color_temperature[i]
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
                                'class_id': bboxes[b]['class_id']
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

    if display is True:
        
        if bboxes is not None:
            # set fixed colors per class using the tab10 color map
            cm = plt.get_cmap('jet') 
            max_number_of_classes = 10
            cNorm  = colors.Normalize(vmin=0, vmax=max_number_of_classes-1)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        
        # show original image
        print('\nAugmenting', image_filename.split('/')[-1], end='')
        print(' [x', end='')
        print(how_many, end='')
        if flip_lr=='all': print(' x2', end='')
        if flip_ud=='all': print(' x2', end='')
        if enhance=='all': print(' x2', end='')
        print(']')
        plt.figure()
        plt.imshow(image, vmin=0, vmax=255)
        plt.title('Original')
        plt.axis('off')
        plt.tight_layout(True)
        if bboxes is not None:
            for bbox in bboxes:
                cls_id = int(bbox['class_id'])
                color_val = scalarMap.to_rgba(cls_id)[0:3]  # keep only rgb, discard a
                plt.gca().add_patch(
                    Rectangle(
                        xy=(bbox['left'],bbox['top']),
                        width=bbox['width'],
                        height=bbox['height'],
                        linewidth=2,
                        edgecolor=color_val,
                        facecolor='none'
                        )
                    )
                plt.gca().text(
                    bbox['left'],
                    bbox['top'] - 2,
                    "{:s}".format(str(bbox['class_id'])),
                    bbox=dict(facecolor=color_val, alpha=0.5),
                    fontsize=8,
                    color="white",
                )
        plt.show()
        
        
        for i,image_transformed in enumerate(dc_augm['Images']):
        
            print('\nTransformation for augmentation', i+1)
            print('Scale:', 
                  dc_augm['Transformations'][i]['Scale'])
            print('Translation_x:', 
                  dc_augm['Transformations'][i]['Translation'][0])
            print('Translation_y:', 
                  dc_augm['Transformations'][i]['Translation'][1])
            print('Rotation:', 
                  dc_augm['Transformations'][i]['Rotation'])
            print('Sheer:', 
                  dc_augm['Transformations'][i]['Sheer'])
            print('Noise:', 
                  dc_augm['Transformations'][i]['Noise'])
            print('Brightness:', 
                  dc_augm['Transformations'][i]['Brightness'])
            print('Colorfullness:', 
                  dc_augm['Transformations'][i]['Colorfullness'])
            print('Color Temperature:', 
                  dc_augm['Transformations'][i]['Color_Temperature'])
            print('Enhance:', 
                  dc_augm['Transformations'][i]['Enhance'])
            print('Flip left->right: ', 
                  dc_augm['Transformations'][i]['Flip_lr'])
            print('Flip up->down: ', 
                  dc_augm['Transformations'][i]['Flip_ud'])
            
            
            plt.figure()
            plt.imshow(image_transformed, interpolation='bilinear')
            plt.title('Augmented #' + str(i+1))
            # plt.tight_layout(True)
            plt.axis('off')
            
            if bboxes is not None:
                for bbox in dc_augm['bboxes'][i]:
                    cls_id = int(bbox['class_id'])
                    color_val = scalarMap.to_rgba(cls_id)[0:3]  # keep only rgb, discard a
                    plt.gca().add_patch(
                    Rectangle(
                        xy=(bbox['left'],bbox['top']),
                        width=bbox['width'],
                        height=bbox['height'],
                        linewidth=2,
                        edgecolor=color_val,
                        facecolor='none'
                        )
                    )
                    plt.gca().text(
                        bbox['left'],
                        bbox['top'] - 2,
                        "{:s}".format(str(bbox['class_id'])),
                        bbox=dict(facecolor=color_val, alpha=0.5),
                        fontsize=8,
                        color="white",
                    )
            plt.show()
    
    
    if bboxes is None: 
        del dc_augm['bboxes']
        del dc_augm['bboxes_discarded']
    
    return dc_augm




