# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 02:49:57 2019

@author: Amjad
"""

from matplotlib import pyplot
from matplotlib.patches import Rectangle
import mtcnn
from mtcnn.mtcnn import MTCNN
import cv2

print("MTCNN version:")
print(mtcnn.__version__)

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
	# show the plot
	pyplot.show()


filename = "test1.png"
#filename = "test2.png"
#filename = "test3.png"

print("Going to load the image")
pixels = cv2.imread(filename)
#pixels = pyplot.imread(filename)

print("Going to load mtcnn model with default/pre-trained weights")
detector = MTCNN()

print("Going to detect faces")
faces = detector.detect_faces(pixels)

for face in faces:
    print(face)

print("Going to draw squares over faces in the image")
draw_image_with_boxes(filename, faces)
detector.save("model.h5")