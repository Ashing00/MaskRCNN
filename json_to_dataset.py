#!/usr/bin/python
# -*- coding: UTF-8 -*-
#!H:\Anaconda3\envs\new_labelme\python.exe
#在cmd中输入python json_to_dateset.py  /path/你的json文件夹的路径
import argparse
import json
import os
import os.path as osp
import base64
import warnings
 
import PIL.Image
import yaml
 
from labelme import utils
 
import cv2
import numpy as np
from skimage import img_as_ubyte
import time
# from sys import argv
 
def main():
	warnings.warn("This script is aimed to demonstrate how to convert the\n"
				  "JSON file to a single image dataset, and not to handle\n"
				  "multiple JSON files to generate a real-use dataset.")
 
	parser = argparse.ArgumentParser()
	parser.add_argument('json_file')
	parser.add_argument('-o', '--out', default=None)
	args = parser.parse_args()
 
	json_file = args.json_file
 
	#freedom
	list_path = os.listdir(json_file)
	print('freedom =', json_file)
	for i in range(0,len(list_path)):
		print("Number :",i)
		start_time = time.time()
		path = os.path.join(json_file,list_path[i])
		if os.path.isfile(path):
 
			data = json.load(open(path))
			img = utils.img_b64_to_arr(data['imageData'])

			label_name_to_value = {'_background_': 0}
			for shape in data['shapes']:
				label_name = shape['label']
				if label_name in label_name_to_value:
					label_value = label_name_to_value[label_name]
				else:
					label_value = len(label_name_to_value)
					label_name_to_value[label_name] = label_value
			
			# label_values must be dense
			label_values, label_names = [], []
			for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
				label_values.append(lv)
				label_names.append(ln)
			assert label_values == list(range(len(label_values)))
			
			lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
			
			captions = ['{}: {}'.format(lv, ln)
				for ln, lv in label_name_to_value.items()]
			print("captions=",captions)
			#ashing lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
 
			#ashing captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
			
			lbl_viz = utils.draw_label(lbl, img, captions)
			out_dir = osp.basename(path).replace('.', '_')
			save_file_name = out_dir
			out_dir = osp.join(osp.dirname(path), out_dir)
 
			if not osp.exists(json_file + '//' + 'labelme_json'):
				os.mkdir(json_file + '//' + 'labelme_json')
			labelme_json = json_file + '//' + 'labelme_json'
 
			out_dir1 = labelme_json + '//' + save_file_name
			if not osp.exists(out_dir1):
				os.mkdir(out_dir1)
 
			#Ashing PIL.Image.fromarray(img).save(out_dir1+'//'+save_file_name+'_img.png')
			PIL.Image.fromarray(img).save(out_dir1+'//'+'img.png') #Ashing 
			#ashing PIL.Image.fromarray(lbl).save(out_dir1+'//'+save_file_name+'_label.png') 
			utils.lblsave(out_dir1+'//'+save_file_name+'_label.png', lbl) #ashing 

			PIL.Image.fromarray(lbl_viz).save(out_dir1+'//'+save_file_name+
			'_label_viz.png')
 
			if not osp.exists(json_file + '//' + 'cv2_mask'):
				os.mkdir(json_file + '//' + 'cv2_mask')
			mask_save2png_path = json_file + '//' + 'cv2_mask'
			################################
			#mask_pic = cv2.imread(out_dir1+'//'+save_file_name+'_label.png',)
			#print('pic1_deep:',mask_pic.dtype)
 
			mask_dst = img_as_ubyte(lbl)  #mask_pic
			#print('pic2_deep:',mask_dst.dtype)
			cv2.imwrite(mask_save2png_path+'//'+save_file_name+'_label.png',mask_dst)
			

			##################################
 
			with open(osp.join(out_dir1, 'label_names.txt'), 'w') as f:
				for lbl_name in label_names:
					f.write(lbl_name + '\n')
 
			warnings.warn('info.yaml is being replaced by label_names.txt')
			info = dict(label_names=label_names)
			with open(osp.join(out_dir1, 'info.yaml'), 'w') as f:
				yaml.safe_dump(info, f, default_flow_style=False)
 
			print('Saved to: %s' % out_dir1)
			print("--- %s seconds ---" % (time.time() - start_time))
 
if __name__ == '__main__':
	#base64path = argv[1]
	main()
