#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import yaml
from PIL import Image


#不使用GPU
# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
# set_session(tf.Session(config=config))


#ashing ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("/home/ashing/pycode/MaskRCNN/Mask_RCNN-master/")
sys.path.append(ROOT_DIR)  

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

# model保存位址
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
iter_num = 0

# trained weights保存位址
COCO_MODEL_PATH = os.path.join("mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)


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

class DrugDataset(utils.Dataset):
	# 得到圖片中有多少物體
	def get_obj_index(self, image):
		n = np.max(image)
		return n

	# 解析labelme中得到的yaml文件，取得mask每一對應的標籤
	def from_yaml_get_class(self, image_id):
		info = self.image_info[image_id]
		with open(info['yaml_path']) as f:
			#temp = yaml.load(f.read())
			temp = yaml.load(f.read(), Loader=yaml.FullLoader)
			labels = temp['label_names']
			del labels[0]
		return labels

	# 重寫draw_mask
	def draw_mask(self, num_obj, mask, image, image_id):
		info = self.image_info[image_id]
		for index in range(num_obj):
			for i in range(info['width']):
				for j in range(info['height']):
					at_pixel = image.getpixel((i, j))
					if at_pixel == index + 1:
						mask[j, i, index] = 1
		return mask

	# 重寫load_shapes，裡面有自己的類別
	# 並在self.image_info訊息中增加path、mask_path 、yaml_path

	def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
		"""Generate the requested number of synthetic images.
		count: number of images to generate.
		height, width: the size of the generated images.
		"""
		# Add classes

	###############################################
	#####									  #####
		
		self.add_class("mold", 1, "NG") 

		
	#####									  #####
	############################################### 
		
		for i in range(count):
			# 讀取圖片長跟寬
			filestr = imglist[i].split(".")[0] 
			#AShing mask_path = mask_floder + "/" + filestr + ".png" 
			mask_path = mask_floder + "/" + filestr + "_json_label.png" #AShing 
			yaml_path = dataset_root_path + "/labelme_json/" + filestr + "_json/info.yaml"		 
			cv_img = cv2.imread(dataset_root_path + "/labelme_json/" + filestr + "_json/img.png")			
			self.add_image("mold", image_id=i, path=img_floder + "/" + imglist[i],
							width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
	# 重寫load_mask
	def load_mask(self, image_id):
		"""Generate instance masks for shapes of the given image ID.
		"""
		global iter_num
		#print("image_id", image_id)
		info = self.image_info[image_id]
		count = 1  # number of object
		img = Image.open(info['mask_path'])
		num_obj = self.get_obj_index(img)
		mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)	   
		mask = self.draw_mask(num_obj, mask, img, image_id)
		occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
		for i in range(count - 2, -1, -1):
			mask[:, :, i] = mask[:, :, i] * occlusion
			occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
		labels = []
		labels = self.from_yaml_get_class(image_id)
		labels_form = []
		
	###############################################
	#####									  #####		   

		
		for i in range(len(labels)):
			if labels[i].find("NG") != -1:
				# print "box"
				labels_form.append("NG")



				
	#####									  #####
	############################################### 
				

		class_ids = np.array([self.class_names.index(s) for s in labels_form])
		return mask, class_ids.astype(np.int32)
	

def get_ax(rows=1, cols=1, size=8):
	"""Return a Matplotlib Axes array to be used in
	all visualizations in the notebook. Provide a
	central point to control graph sizes.

	Change the default size attribute to control the size
	of rendered images
	"""
	_, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
	return ax

def train_model():
	dataset_root_path = r"train_data_mold"
	img_floder = os.path.join(dataset_root_path, "pic")
	mask_floder = os.path.join(dataset_root_path, "cv2_mask")
	
	# yaml_floder = dataset_root_path
	imglist = os.listdir(img_floder) 
	#print("imglist=",imglist)
	count = len(imglist)   
	#ashing imglist.remove('.ipynb_checkpoints');  
	print("img number :",len(imglist))
	
	# train與val資料集
	dataset_train = DrugDataset()
	dataset_train.load_shapes(count-1, img_floder, mask_floder, imglist, dataset_root_path)
	dataset_train.prepare()


	#print(imglist, dataset_root_path)
	dataset_val = DrugDataset()
	dataset_val.load_shapes(count-1, img_floder, mask_floder, imglist, dataset_root_path)
	dataset_val.prepare()	

	# Create models in training mode
	config = BreadConfig()
	config.display()
	model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

	###############################################
	#####									  ##### 
	

	# 第一次訓練，請填coco，在產生訓練後的模型後，如果想繼續沿用請改成last
	init_with = "coco"	# imagenet, coco, or last
	
	
	#####									  #####
	############################################### 
   
	if init_with == "imagenet":
		model.load_weights(model.get_imagenet_weights(), by_name=True)
	elif init_with == "coco":
		# Load weights trained on MS COCO, but skip layers that
		# are different due to the different number of classes
		# See README for instructions to download the COCO weights
	
		model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
	 
	elif init_with == "last":
		# Load the last models you trained and continue training
		checkpoint_file = model.find_last()
		model.load_weights(checkpoint_file, by_name=True)

	# Train the head branches
	# Passing layers="heads" freezes all layers except the head
	# layers. You can also pass a regular expression to select
	# which layers to train by name pattern.
	
	
	
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=5,
				layers='heads')

	# Fine tune all layers
	# Passing layers="all" trains all layers. You can also
	# pass a regular expression to select which layers to
	# train by name pattern.
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=100,
				layers="all")

train_model()

