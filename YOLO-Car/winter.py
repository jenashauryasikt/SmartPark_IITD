#importing required libraries
import numpy as np
import cv2
import time
import sys
import os
import matplotlib.pyplot as plt
import urllib.request
import requests
from io import BytesIO
from datetime import datetime
from PIL import Image


#files for car_detection
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

net1 = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

target = open("parking.txt", "r")
total = target.readline()
total = int(total,10)
target.close()

def park(image):
	
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
	(H, W) = image.shape[:2]
	
	ln = net1.getLayerNames()
	ln = [ln[i[0] - 1] for i in net1.getUnconnectedOutLayers()]
	
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net1.setInput(blob)
	layerOutputs = net1.forward(ln)
	
	boxes = []
	confidences = []
	classIDs = []
	threshold = 0.5
	
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > threshold:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)
	car = 0

	if len(idxs) > 0:

		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			if(LABELS[classIDs[i]]=='car'):
				car+=1
				color = (0,255,0)
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
				text = "{}".format(LABELS[classIDs[i]])
				cv2.putText(image, text, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, color, 1)

	text1 = "No. of spots filled: " + str(total-car) + "/" + str(total)
	text2 = "No. of spots available: " + str(car) + "/" + str(total)
	streamer = datetime.now().strftime("%d/%m/%Y::%H:%M:%S")
	color1 = (0,0,255)
	color2 = (0,255,0)
	color3 = (0,255,255)

	cv2.putText(image, text1, (4,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2)
	cv2.putText(image, text2, (4,60), cv2.FONT_HERSHEY_SIMPLEX, 1, color2, 2)
	cv2.putText(image, streamer, (4,1000), cv2.FONT_HERSHEY_SIMPLEX, 1, color3, 2)
	return image, (total-car)


# car detection for streaming
def checker(url):

	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	a = time.time()	
	while(True):
		t = time.time()
		video_name_time = datetime.now().strftime('%Y%m%d%H%M')
		video_name = "Park" + video_name_time + ".mp4"
		out = cv2.VideoWriter(video_name, fourcc, 2.0, (624, 416))
		while True:
			s = time.time()
			response = requests.get(url)
			img = Image.open(BytesIO(response.content))
			img = np.asarray(img)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			img1, spots = park(img)
			op = cv2.resize(img1, (624, 416))

			out.write(op)
			print(time.time()-s)
			print("Spots available: ", spots)
			print("\n")	
			cv2.imwrite('park_now.jpg', op)
			if((time.time()-t)//60==1):
				break

		try:
			if((time.time()-a)//180==1):
				dir_name = os.getcwd()
				test = os.listdir(dir_name)

				for item in test:
				    if (item.startswith("Park") and item.endswith(".mp4")):
				        os.remove(os.path.join(dir_name, item))

				a = time.time()
			else:
				continue
		except:
			print('Access error')
			a = time.time()

def checker2(video):

	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	a = time.time()	
	while(True):
		t = time.time()
		video_name_time = datetime.now().strftime('%Y%m%d%H%M')
		video_name = "Park" + video_name_time + ".mp4"
		out = cv2.VideoWriter(video_name, fourcc, 2.0, (624, 416))
		cap = cv2.VideoCapture(video)
		cap.open(video)
		while True:
			s = time.time()
			ret, frame = cap.read()
			if ret == False:
				break

			image, spots = park(frame)

			image = cv2.resize(image, (624, 416))
			out.write(image)
			print(time.time()-s)
			print("Spots available: ", spots)
			print("\n")	
			cv2.imwrite('park_now.jpg', image)
			if((time.time()-t)//60==1):
				break

		try:
			if((time.time()-a)//180==1):
				dir_name = os.getcwd()
				test = os.listdir(dir_name)

				for item in test:
				   if (item.startswith("Park") and item.endswith(".mp4")):
				        os.remove(os.path.join(dir_name, item))

				a = time.time()
			else:
				continue
		except:
			print('Access error')
			a = time.time()

# url to be given for live streaming
# url to be given for live streaming
target = open("address.txt", "r")
ip = target.readline()
target.close()

target = open("port.txt", "r")
port = target.readline()
target.close()

target = open("user.txt", "r")
user = target.readline()
target.close()

target = open("passwd.txt", "r")
pwd = target.readline()
target.close()

target = open("relay.txt", "r")
relay = target.readline()
target.close()

url = "rtsp://" + str(user) + ":" + str(pwd) + "@" + str(ip) + ":" + str(port) + "/" + str(relay)

video = "test1.mp4"

target = open("LiveOrNot.txt", "r")
decision = target.readline()
target.close() 

if __name__ == '__main__': 

	if decision == "yes":
		checker(url)

	if decision == "no":
		checker2(video)