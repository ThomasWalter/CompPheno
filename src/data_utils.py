import os
import numpy as np
from skimage.io import imread
import csv

from sklearn.model_selection import train_test_split

#WORK_DIRECTORY = 'data/mnist-data'
IN_FOLDER = './data/classifier/Classifier_full_2015_06_04_b'
IMAGE_FOLDER = os.path.join(IN_FOLDER, 'samples')
#data = data.reshape(num_images, image_size, image_size, 1)

def read_class_definition(in_folder=IN_FOLDER):

	fp = open(os.path.join(in_folder, 'class_definition.txt'), 'r')
	reader = csv.reader(fp, delimiter='\t')
	name_to_label = {}
	label_to_name = {}
	lut = {}
	for label, class_name, class_color in reader:
		name_to_label[class_name] = int(label)
		label_to_name[int(label)] = class_name
		lut[int(label)] = class_color

	class_definition = {
    	'name_to_label': name_to_label,
    	'label_to_name': label_to_name,
    	'colors': lut
    }

	return class_definition

def read_image_data_from_sample_folder(in_folder=IN_FOLDER):
	image_folder = os.path.join(in_folder, 'samples')

	#class_definition = read_class_definition(in_folder)

	images = {}
	output = {}
	for y_str in filter(lambda x: os.path.isdir(os.path.join(image_folder, x)), os.listdir(image_folder)):
		filenames = list(filter(lambda x: x.endswith('.png'), os.listdir(os.path.join(image_folder, y_str))))
		for filename in filenames:
			images[os.path.splitext(filename)[0]] = imread(os.path.join(image_folder, y_str, filename))
			output[os.path.splitext(filename)[0]] = y_str

	#X = np.array(image_list)
	#y_vec = np.array([class_definition['name_to_label'][y] for y in y_list])

	return images, output

def read_image_data(in_folder=IN_FOLDER):

	class_definition = read_class_definition(in_folder)

	images, output = read_image_data_from_sample_folder(in_folder)

	sample_filename = os.path.join(in_folder, 'data', 'features.samples.txt')
	fp = open(sample_filename, 'r')
	lines = list(fp.readlines())
	filenames = [x.split('\t')[1].strip('\n') for x in lines]
	y_list = [x.split('\t')[0] for x in lines]
	fp.close()

	X = np.array([images[x + '____img'] for x in filenames])
	yvec = np.array([class_definition['name_to_label'][y] for y in y_list])
	
	return X, yvec


