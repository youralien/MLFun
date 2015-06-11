from easynet.easynet import analyze

import os
import sys
import numpy as np

def cache_image_features(img_dir, dest_dir):

	if not os.path.exists(dest_dir):
		# FIX: mkdir not working
		os.mkdir(dest_dir)

	fns = os.listdir(img_dir)
	for i, fn in enumerate(fns):
		sys.stdout.write(str(i))
		features = analyze(img_dir + fn)
		fn_w_out_extension = fn.split('.')[0]
		np.save(dest_dir+fn_w_out_extension, features)
		sys.stdout.flush()

def cache_train():
	cache_image_features(
		  "/home/luke/datasets/coco/train2014/"
		, "/home/luke/datasets/coco/features/train2014/")

def cache_valid():
	cache_image_features(
		  "/home/luke/datasets/coco/val2014/"
		, "/home/luke/datasets/coco/features/val2014/")

if __name__ == '__main__':
	import ipdb
	ipdb.set_trace()