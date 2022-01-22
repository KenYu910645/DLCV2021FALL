# Python import 
import glob
import os
import numpy as np
import math 
import pandas as pd 
import time
import collections
import shutil
# Sitk
import SimpleITK as sitk
# Pytorch
import torch
import torch.cuda
from torch.utils.data import Dataset
# Skimage, Image processing
import skimage
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from skimage import morphology
from skimage import measure
from skimage.transform import resize
# Pillow
from PIL import Image, ImageDraw
# Scipy
from scipy import ndimage as ndi
# Matplotlib
import matplotlib.pyplot as plt # pip install matplotlib==2.2.3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import time 

HIGH_LIMIT = 1000
LOW_LIMIT = -1000
THRESHOLD = -400 # Unit HU #[-800 # -700 # -600 # -500, -400. -300]  this is good 

df_node = pd.read_csv("data/annotations.csv")

def get_segmented_lungs(im):
    # Reference : https://www.kaggle.com/arnavkj95/candidate-generation-and-luna16-preprocessing/notebook
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < THRESHOLD 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2) # This is kind of weird
    binary = binary_erosion(binary, selem)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = LOW_LIMIT
    return im

# There's a very big nodule on the lung, but been segmented out 
# 1.3.6.1.4.1.14519.5.2.1.6279.6001.624425075947752229712087113746
# bad noise and big tumor
# 1.3.6.1.4.1.14519.5.2.1.6279.6001.121391737347333465796214915391
VIZ = False # whether to output viz result
DISPLAY = False # if wanna to display image to screen 
DELETE_GT_PROP = False # Switch to True if wanna output candidate.csv for training
APPEND_GT = False # Switch to True if wanna train 
INPUT_DIR = "data/train/"
OUTPUT_DIR = "data/train_seg_lung_tmp/"

mhd_list = glob.glob(INPUT_DIR + "*.mhd")
for mhd_idx, fn in enumerate(mhd_list):
    t_start = time.time()
    series_uid = os.path.split(fn)[1][:-4]
    # Load image
    ct_mhd = sitk.ReadImage( fn )
    ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
    ct_a.clip(LOW_LIMIT, HIGH_LIMIT, ct_a)

    # Get origin spacing 
    origin = np.array(ct_mhd.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(ct_mhd.GetSpacing())    # spacing of voxels in world coor. (mm)
    

    # New numpy array for lung and nodule segmented
    # lung_a   = np.zeros(ct_a.shape)
    lung_a   = ct_a.copy()
    # 
    N_depth = ct_a.shape[0]
    n_col = math.floor(math.sqrt(N_depth))
    n_row = math.ceil(N_depth/n_col)

    
    t_start = time.time()
    for idx in range(N_depth):
        # segment lung
        lung_a[idx] = get_segmented_lungs(ct_a[idx])
    
    print(f"Finish Segment operation. {round(time.time() - t_start, 2)} sec.")
    
    t_start = time.time()
    np.save(OUTPUT_DIR + series_uid + ".npy", lung_a)
    print(f"Finish saving operation. {round(time.time() - t_start, 2)} sec.")
    
    # sitk.WriteImage(lung_a, series_uid + ".mhd")
    print(f"Processed {mhd_idx+1}/{len(mhd_list)}")

