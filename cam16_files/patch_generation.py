# Install the OpenSlide C library and Python bindings
!apt-get install openslide-tools
!apt-get install python3-openslide

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import shutil
import time
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from PIL import Image
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split

!pip install ipython-autotime

# %load_ext autotime
# %tensorflow_version 2.x
import tensorflow as tf

# Mount google drive
from google.colab import drive
drive.mount('/content/drive/')

# Get the slides and masks path
import re

# # Path to normal (training) slides
normal_dir = "/content/drive/MyDrive/Cam16_images"
# # Path to mask tif files
masks_dir = "/content/drive/Shareddrives/271B_project/masks"
# # Path to test mask tif files
test_masks_dir = "/content/drive/Shareddrives/271B_project/test_masks"
# # Path to tumor (training) slides
tumor_dir = "/content/drive/MyDrive/CAM16"
# Path to test slides
test_dir = "/content/drive/MyDrive/cam_16_test"

list_files = os.listdir(normal_dir)
list_files_2 = os.listdir(masks_dir)
list_files_3 = os.listdir(tumor_dir)
list_files_4 = os.listdir(test_dir)
list_files_5 = os.listdir(test_masks_dir)

def filter_path(file_path):
    if re.findall("[0-9].tif", file_path):
        # Slide
        return 1
    elif re.findall("_mask.tif", file_path):
        # Mask
        return 2
    else:
        return 0

name_normal = list(filter(lambda x: filter_path(x) == 1, list_files))
name_masks = list(filter(lambda x: filter_path(x) == 2, list_files_2))
name_tumor = list(filter(lambda x: filter_path(x) == 1, list_files_3))
name_test = list(filter(lambda x: filter_path(x) == 1, list_files_4))
name_test_masks = list(filter(lambda x: filter_path(x) == 2, list_files_5))

tumor_slides = []
normal_slides = []
test_slides = []

for x in name_tumor:
  if x[0:5] == 'tumor':
    tumor_slides.append(x)

for x in name_normal:
  if x[0:6] == 'normal':
    normal_slides.append(x)

for x in name_test:
  if x[0:4] == 'test':
    test_slides.append(x)

path_test = []
path_tumor = []
path_masks = []
path_test_masks = []
path_normal = []

for name_slide in normal_slides:
  
    path_normal.append(os.path.join(normal_dir, name_slide))

for name_slide in tumor_slides:

    name_mask = name_slide[:-4] + "_mask.tif"
    path_tumor.append(os.path.join(tumor_dir, name_slide))

    if os.path.isfile(os.path.join(masks_dir, name_mask))==True:
      path_masks.append(os.path.join(masks_dir, name_mask))
    
    assert name_slide == (name_mask[:-9] + ".tif")

for name_slide in test_slides:

    name_mask = name_slide[:-4] + "_mask.tif"
    path_test.append(os.path.join(test_dir, name_slide))

    if os.path.isfile(os.path.join(test_masks_dir, name_mask))==True:
      path_test_masks.append(os.path.join(test_masks_dir, name_mask))
    
    assert name_slide == (name_mask[:-9] + ".tif")
    # assert name_mask in name_masks

#Check the paths

print(path_test)
print(path_test_masks)
print(path_masks)

# Read a region from the slide and return a numpy RBG array

from skimage.color import rgb2gray

def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

# Make patches for training by sliding a window across slides 
# (Note that for patches without tumor, we just accept those with cells.)
# The criteria are mean standard deviation across color channel and mean intensity after converted to gray.
# This is to save RAM and to make training set informative.)
def make_patch_set(path_slides, path_masks, patch_size=100, level=4):

    slide_patches = {}
    tumor_indicator = {}
    threshold_std = 5
    # threshold_intensity = 0.2
    dict_patches = {}

    for slide_path in path_slides:

      counter = 0
      count = 0
      slide_whole = open_slide(slide_path)
      tumor_mask = open_slide(slide_path)
      temp_name = os.path.basename(slide_path)[0:-4]
      name_mask = temp_name + "_mask.tif"
      key = int(temp_name[-3:])

      if os.path.isfile(os.path.join(test_masks_dir, name_mask))==True:
        counter = 1
        tumor_mask = open_slide(os.path.join(test_masks_dir, name_mask))

      n_step_width = slide_whole.level_dimensions[level][0] // patch_size
      n_step_height = slide_whole.level_dimensions[level][1] // patch_size
      downsample_factor = slide_whole.level_downsamples[level]  

      for step_width in range(n_step_width):
        for step_height in range(n_step_height):
            slide_window = [int(patch_size * step_width * downsample_factor), int(patch_size * step_height * downsample_factor)] 

            # Slide
            slide_patch = read_slide(slide_whole, 
                                  x=slide_window[0], 
                                  y=slide_window[1], 
                                  level=level, 
                                  width=patch_size, 
                                  height=patch_size)   
                      
            # Mask
            if counter == 1:
              im_mask = read_slide(tumor_mask, 
                                  x=slide_window[0], 
                                  y=slide_window[1], 
                                  level=level, 
                                  width=patch_size, 
                                  height=patch_size) 
            
            # Only save those patches that contain cells
            if np.mean(np.std(slide_patch, axis=-1)) > threshold_std:

                if key in slide_patches:
                  slide_patches[key].append(slide_patch)
                  count += 1
                else:
                  slide_patches[key] = [slide_patch]
                  count += 1

                if counter == 1:  
                  if np.sum(im_mask[:,:,0]) > 0:
                    if key in tumor_indicator:
                      tumor_indicator[key].append(1)
                    else:
                      tumor_indicator[key] = [1]
                  else:
                    if key in tumor_indicator:
                      tumor_indicator[key].append(0)
                    else:
                      tumor_indicator[key] = [0]
                else:
                  if key in tumor_indicator:
                    tumor_indicator[key].append(0)
                  else:
                    tumor_indicator[key] = [0]
              
      dict_patches[key] = count

    return slide_patches, tumor_indicator, dict_patches

# Make patches for training and testing by sliding a window across slides (for normal and test slides only) 
# (Note that for patches without tumor, we just accept those with cells.)
# The criteria are mean standard deviation across color channel and mean intensity after converted to gray.
# This is to save RAM and to make training set informative.)

def make_patch_set_all(path_slides, patch_size=100, level=4):
    slide_patches = []
    tumor_indicator = []
    threshold_std = 5
    threshold_intensity = 0.2

    for slide_path in path_slides:
      slide_whole = open_slide(slide_path)
      n_step_width = slide_whole.level_dimensions[level][0] // patch_size
      n_step_height = slide_whole.level_dimensions[level][1] // patch_size
      downsample_factor = slide_whole.level_downsamples[level]                 
      for step_width in range(n_step_width):
        for step_height in range(n_step_height):
            slide_window = [int(patch_size * step_width * downsample_factor), int(patch_size * step_height * downsample_factor)] 

            # Slide
            slide_patch = read_slide(slide_whole, 
                                  x=slide_window[0], 
                                  y=slide_window[1], 
                                  level=level, 
                                  width=patch_size, 
                                  height=patch_size)               
            
            # Only save those patches that contain cells
            if np.mean(np.std(slide_patch, axis=-1)) > threshold_std:
                slide_patches.append(slide_patch)
                tumor_indicator.append(0)  

    return slide_patches, tumor_indicator

# Extract the patches and labels
level = 4
patch_size = 128

# # Save the unbalanced training datasets and testing datasets     
variable_folder =  "/content/drive/MyDrive/CAM16_patches"
if os.path.isdir(variable_folder)==False:
    os.mkdir(variable_folder)

file_name_1 = "unbalanced_training_tumor_dataset" + "_level" + str(level) + "_size" +  str(patch_size) + ".pkl"
file_name_2 = "testing_dataset_with_gt" + "_level" + str(level) + "_size" +  str(patch_size) + ".pkl"

# # Create patches pertaining to the test dataset and save them in a pickle file
test_patches, tumor_indicator_test, dict_patches = make_patch_set(path_test, path_test_masks, patch_size=patch_size, level=level)

with open(os.path.join(variable_folder, file_name_2), 'wb') as f:
    pickle.dump([test_patches, tumor_indicator_test, dict_patches], f, protocol=-1)

# # Create patches pertaining to the tumor dataset and save them in a pickle file
tumor_patches, tumor_indicator_tumor, dict_patches = make_patch_set(path_tumor[2:], path_masks, patch_size=patch_size, level=level)

with open(os.path.join(variable_folder, file_name_1), 'wb') as f:
    pickle.dump([tumor_patches, tumor_indicator_tumor, dict_patches], f, protocol=-1)

# # Create patches pertaining to the normal dataset and save them in a pickle file
file_name_3 = "normal_dataset" + "_level" + str(level) + "_size" +  str(patch_size) + ".pkl"
normal_patches, tumor_indicator_normal = make_patch_set_all(path_normal, patch_size=patch_size, level=level)

with open(os.path.join(variable_folder, file_name_3), 'wb') as f:
    pickle.dump([normal_patches, tumor_indicator_normal], f, protocol=-1)

# Compute the number of patches with tumor in the unbalanced tumor training set
n_patch = len(tumor_indicator_tumor)
n_patch_tumor = sum(tumor_indicator_tumor)
print("Percent of patches with tumor: {:.1f}% ({:d}/{:d})".format(n_patch_tumor/n_patch*100, n_patch_tumor, n_patch))

# Segregating tumor training set patches into patches with tumor and without tumor
slide_patches_tumor = [tumor_patches[i] for i in range(n_patch) if tumor_indicator_tumor[i]==1]
slide_patches_notumor = [tumor_patches[i] for i in range(n_patch) if tumor_indicator_tumor[i]==0]

n_patches_no_tumor = n_patch - n_patch_tumor
n_patches_per_slide = int(n_patches_no_tumor / len(path_tumor))
n_patches_tumor_per_slide = int(n_patch_tumor / len(path_tumor)) + 1

# Create a balanced dataset

from skimage.color import rgb2gray

counter = 0
slide_patches_notumor_cell = []
threshold_std = 5
threshold_intensity = 0.2
for ind_slide in range(len(path_tumor)):
    if counter == 1:
      break
    count = 0
    for ind_patch in range(n_patches_per_slide):
        ind_select = ind_patch + ind_slide * n_patches_per_slide
        if ind_select % 2 == 0:
          im_patch = slide_patches_notumor[ind_select]
        else:
          im_patch = normal_patches[ind_select]
        im_gray = rgb2gray(im_patch)
        if np.mean(np.std(im_patch, axis=-1)) > threshold_std and np.mean(im_gray) > threshold_intensity:
            slide_patches_notumor_cell.append(im_patch)
            count += 1

        if len(slide_patches_notumor_cell) == len(slide_patches_tumor):
          counter = 1
          break

        if count == n_patches_tumor_per_slide or ind_select == n_patches_no_tumor-1:
            break

dataset_patches = slide_patches_tumor + slide_patches_notumor_cell
dataset_label = np.hstack((np.ones((n_patch_tumor,)), np.zeros((n_patch_tumor,))))

# Save the balanced dataset    
with open(os.path.join(variable_folder, "balanced_dataset_level_4_size_64.pkl"), 'wb') as f:
    pickle.dump([dataset_patches, dataset_label], f, protocol=-1)

# Check the patches without tumor to see if they contain cells
index_shuffle = np.arange(len(slide_patches_notumor_cell))
np.random.shuffle(index_shuffle)
dim_axis = [20, 7]
fig = plt.figure(figsize=(24,10))
for i in range(dim_axis[1]):
    for j in range(dim_axis[0]):
        ax = fig.add_subplot(dim_axis[1], dim_axis[0], i*dim_axis[0]+j+1)
        plt.imshow(slide_patches_notumor_cell[index_shuffle[i*dim_axis[0]+j]])
        im_gray = rgb2gray(slide_patches_notumor_cell[index_shuffle[i*dim_axis[0]+j]])
        std_cross_channel = np.std(slide_patches_notumor_cell[index_shuffle[i*dim_axis[0]+j]], axis=-1)
        # ax.set_title(str(round(np.mean(im_gray), 2)) + ", " + str(round(np.mean(std_cross_channel), 2)))
        ax.set_title(str(round(np.mean(std_cross_channel), 2)))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

# Check the patches with tumor to see if they contain cells
index_shuffle = np.arange(len(slide_patches_tumor))
np.random.shuffle(index_shuffle)
dim_axis = [20, 7]
fig = plt.figure(figsize=(24,10))
for i in range(dim_axis[1]):
    for j in range(dim_axis[0]):
        ax = fig.add_subplot(dim_axis[1], dim_axis[0], i*dim_axis[0]+j+1)
        plt.imshow(slide_patches_tumor[index_shuffle[i*dim_axis[0]+j]])
        im_gray = rgb2gray(slide_patches_tumor[index_shuffle[i*dim_axis[0]+j]])
        std_cross_channel = np.std(slide_patches_tumor[index_shuffle[i*dim_axis[0]+j]], axis=-1)
        # ax.set_title(str(round(np.mean(im_gray), 2)) + ", " + str(round(np.mean(std_cross_channel), 2)))
        ax.set_title(str(round(np.mean(std_cross_channel), 2)))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)