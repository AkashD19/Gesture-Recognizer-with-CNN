import cv2, pickle
import numpy as np
from pygame import mixer
import os
from keras.models import load_model
from gtts import gTTS
from tempfile import TemporaryFile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
prediction = None
model = load_model('model.h5')
image = cv2.imread('gestures/train1/100.jpg', 0)
image_x, image_y = image.shape()


def prediction(model, image):
	image = cv2.resize(image, (image_x, image_y))
	image = np.array(image, dtype=np.float32)
	image = np.reshape(image, (1, image_x, image_y, 1))
	prob = model.predict(image)[0]
	problist = list(prob).index(max(prob))
	return max(prob), problist

def retpred(problist):
	if problist==0:
		return "Nikos"	        
	elif problist==1:
		return "project"
	elif problist==2:
		return "our"   
	elif problist==3:
		return "Hi"
	elif problist==4:
		return "is"
	elif problist==5:
		return "This"
	else:
		return

def main():
	global prediction	
	capturecam = cv2.VideoCapture(1)
	if capturecam.read()[0] == False:
		capturecam = cv2.VideoCapture(0)
	with open("histogram1", "rb") as f:
		hist = pickle.load(f)
	x, y, w, h = 300, 100, 300, 300
	text1="  "
	while True:
		text = ""
		image = capturecam.read()[1]
		image = cv2.flip(image, 1)
		HSVimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([HSVimg], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)
		gauss = cv2.GaussianBlur(dst, (11,11), 0)
		gauss = cv2.medianBlur(blur, 15)
		t = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		t = cv2.merge((t,t,t))
		t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
		t = t[y:y+h, x:x+w]
		contours = cv2.findContours(t.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				saveimg = t[y1:y1+h1, x1:x1+w1]
				
				if w1 > h1:
					saveimg = cv2.copyMakeBorder(saveimg, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					saveimg = cv2.copyMakeBorder(saveimg, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				
				prob, problist = prediction(model, saveimg)
				print(problist, prob)
				
				if prob*100 > 80:
					text = retpred(problist)
					if text!=text1:
						text1=text
						tts=gTTS(text=text, lang='en')
						mixer.init()
						sf = TemporaryFile()
						tts.write_to_fp(sf)
						sf.seek(0)
						mixer.music.load(sf)
						mixer.music.play()
							
					print(text)
		showb = np.zeros((480, 640, 3), dtype=np.uint8)
		list_words = text.split(" ")
		length = len(list_words)
		sent = []
		b_index = 0
		e_index = 2
		while length > 0:
			part = ""
			for a in list_words[b_index:e_index]:
				part = part + " " + a
			sent.append(part)
			b_index += 2
			e_index += 2
			length -= 2
		y = 200
		for text in splitted_text:
			cv2.putText(showb, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
			y += 50
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((image, showb))
		cv2.imshow("Recognition", res)
		cv2.imshow("t", t)
		if cv2.waitKey(1) == ord('q'):
			break
cv2.destroyAllWindows()
keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		
main()
