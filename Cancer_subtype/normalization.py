# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import glob
import sys
import os
from Run_StainSep import run_stainsep
from Run_ColorNorm import run_colornorm, run_batch_colornorm
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])

nstains=2    #number of stains
lamb=0.01
op=0 
# 2= color normalization of one image with one target image
# 3= color normalization of all images in a folder with one target image

if op==2:
	level=0
	output_direc="/home/roshan/normalized_images/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)

	source_filename="/home/roshan/TCGA-04-1517-11A-01-TS1-r7-c5-x4098-y6152-w1024-h1025.png"
	target_filename="/home/roshan/TCGA-04-1517-11A-01-TS1-r14-c37-x36881-y13330-w1024-h1024.png"

	if not os.path.exists(source_filename):
		print ("Source file does not exist")
		sys.exit()
	if not os.path.exists(target_filename):
		print ("Target file does not exist")
		sys.exit()
	background_correction = True	
	run_colornorm(source_filename,target_filename,nstains,lamb,output_direc,level,background_correction)
    
elif op==3:
	level=0

	input_direc="/home/roshan/source_images"
	output_direc="/home/roshan/normalized_images/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)
	file_type="*.png"
	target_filename="/home/roshan/TCGA-04-1517-11A-01-TS1-r14-c37-x36881-y13330-w1024-h1024.png"
	if not os.path.exists(target_filename):
		print ("Target file does not exist")
		sys.exit()
	if len(sorted(glob.glob(input_direc+file_type)))==0:
		print ("No source files found")
		sys.exit()
	filenames=[target_filename]+sorted(glob.glob(input_direc+file_type))
	
	background_correction = True
	run_batch_colornorm(filenames,nstains,lamb,output_direc,level,background_correction)

