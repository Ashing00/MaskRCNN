import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time


def Contrast_and_Brightness(alpha, beta, img):
	blank = np.zeros(img.shape, img.dtype)
	# dst = alpha * img + beta * blank
	dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
	return dst

# Root directory of the project
#ROOT_DIR = os.path.abspath("../")
ROOT_DIR = os.path.abspath("/home/ashing/pycode/MaskRCNN/Mask_RCNN-master/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#import coco

class BreadConfig(Config):

	NAME = "mold"

	# Train on 1 GPU and 8 images per GPU. We can put multiple images on each
	# GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# Number of classes (including background)
	NUM_CLASSES = 1+1  # background + 1 class

	# Use small images for faster training. Set the limits of the small side
	# the large side, and that determines the image shape.
	IMAGE_MIN_DIM = 512
	IMAGE_MAX_DIM = 512

	# Use smaller anchors because our image and objects are small
	#RPN_ANCHOR_SCALES = (4*6,8 * 6, 16 * 6, 32 * 6, 64 * 6)  # anchor side in pixels
	RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)


	# Ratios of anchors at each cell (width/height)
	# A value of 1 represents a square anchor, and 0.5 is a wide anchor
	RPN_ANCHOR_RATIOS = [0.5, 1, 2]
	#RPN_ANCHOR_RATIOS = [0.25, 0.5, 1, 2, 4]

	# Reduce training ROIs per image because the images are small and have
	# few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
	TRAIN_ROIS_PER_IMAGE = 64

	# Use a small epoch since the data is simple
	STEPS_PER_EPOCH = 100

	# use small validation steps since the epoch is small
	VALIDATION_STEPS = 1
	
	# ROIs below this threshold are skipped
	DETECTION_MIN_CONFIDENCE = 0.5

	# Non-maximum suppression threshold for detection
	DETECTION_NMS_THRESHOLD = 0.3
	
	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.7


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
test_MODEL_PATH = os.path.join(MODEL_DIR ,"mold20190705T1422/mask_rcnn_mold_0100.h5")
print("test_MODEL_PATH=",test_MODEL_PATH)
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "mold_test")

class InferenceConfig(BreadConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(test_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names =['BG', 'NG']


###=======================================
start_time = time.time()
 # Load a random image from the images folder

file_names = r'/home/ashing/pcb/images/ok.jpg' # next(os.walk(IMAGE_DIR))[2]
#file_names = r'/home/ashing/pycode/MaskRCNN/Mask_RCNN-master/nb_test/NB_33.jpg' # next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#image = skimage.io.imread(file_names)
image = cv2.imread(file_names)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

#image=Contrast_and_Brightness(1.0,-50,image)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print("rois=",r['rois'])
#print("masks=",r['masks'])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
print("--- %s seconds ---" % (time.time() - start_time))
plt.savefig("mold_temp.jpg")

img = cv2.imread("mold_temp.jpg")

cv2.namedWindow('mold_detection', cv2.WINDOW_NORMAL) #WINDOW_AUTOSIZE
cv2.resizeWindow("mold_detection", 1024, 1024)
cv2.imshow('mold_detection', img)


###=======================================


start_time = time.time()
 # Load a random image from the images folder

file_names = r'/home/ashing/pcb/images/test2.jpg' # next(os.walk(IMAGE_DIR))[2]

image = cv2.imread(file_names)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print("rois=",r['rois'])
#print("masks=",r['masks'])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
print("--- %s seconds ---" % (time.time() - start_time))
plt.savefig("mold_temp2.jpg")

img2 = cv2.imread("mold_temp2.jpg")

cv2.namedWindow('mold_detection2', cv2.WINDOW_NORMAL) #WINDOW_AUTOSIZE
cv2.resizeWindow("mold_detection2", 1024, 1024)
cv2.imshow('mold_detection2', img2)

###=======================================


start_time = time.time()
 # Load a random image from the images folder

file_names = r'/home/ashing/pcb/images/test5.jpg' # next(os.walk(IMAGE_DIR))[2]

image = cv2.imread(file_names)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print("rois=",r['rois'])
#print("masks=",r['masks'])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
print("--- %s seconds ---" % (time.time() - start_time))
plt.savefig("mold_temp3.jpg")

img3 = cv2.imread("mold_temp3.jpg")

cv2.namedWindow('mold_detection3', cv2.WINDOW_NORMAL) #WINDOW_AUTOSIZE
cv2.resizeWindow("mold_detection3", 1024, 1024)
cv2.imshow('mold_detection3', img3)



k = cv2.waitKey(0)
if k == 27:			# wait for ESC key to exit
	cv2.destroyAllWindows()











