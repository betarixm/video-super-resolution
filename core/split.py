import cv2
import numpy as np
import os

from PIL import Image

path = './videos/'
fileList = []
NumVideos = 380

for i in range(0, NumVideos):
	cap = cv2.VideoCapture(path + str(i))
	
	try:
		if not os.path.exists('Frame'):
			os.makedirs('Frame')

	except OSError:
		print('Error: Creating directory of data')

	dirPath = './Frame/HR/' + str(i)
	print(dirPath)
	os.makedirs(dirPath)
	LRPath = './Frame/LR/' + str(i)
	os.makedirs(LRPath)

	currentFrame = 0
	while(True):
		ret, frame = cap.read()
		if not ret: break

		name = dirPath + '/' + str(currentFrame).zfill(3) + '.png'
		print('Creating...' + name)
		cv2.imwrite(name, frame)

		HR_img = Image.open(name)
		w, h = HR_img.size
		left = (w-128)/2
		top = (h - 128)/2
		right = (w + 128)/2
		bottom = (h + 128)/2
		area = (left, top, right, bottom)
		cropped_img = HR_img.crop(area)
		HR_img.close()
		cropped_img.save(name)

		resize_img = cropped_img.resize((32, 32))
		resize_img.save(LRPath + '/' + str(currentFrame).zfill(3) + '.png')
		currentFrame += 1

	cv2.destroyAllWindows()
