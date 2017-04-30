import numpy as np
import cv2
import os 
from wand.image import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

video = cv2.VideoCapture(0) 
name = raw_input('What is your name? \n')
os.mkdir('/home/manhthe/project/data/original/'+name)
print("Created folder success!!!\n")
sample_num = 0
while(video.isOpened()):

	ret,frame = video.read()

	

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
 	
	for (x,y,w,h) in faces:
		sample_num = sample_num + 1
		cv2.imwrite("/home/manhthe/project/data/original/%s/" % (name)+str(sample_num)+".jpg",frame[y:y+h,x:x+w])
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		# roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		cv2.waitKey(1) 

	cv2.imshow('Video',frame)
	#cv2.waitKey(1)
	
	if(sample_num>25):
		break

print "Done!!!"

folder_list=[os.path.join(folder_dir,folder) for folder in os.listdir(folder_dir)]

f = os.listdir("/home/trantrunghieu/lv/project/data/original/%s" % name)
i = 0
# os.mkdir(folder_list)
print("Created folder resize success!!!\n")
for fol in os.listdir(folder_dir):
	os.mkdir('/home/manhthe/project/dulieu/'+fol)
	folder = os.path.join(folder_dir,fol)
	print(folder)
	for filename in os.listdir(folder):
		print(filename)
		with Image(filename="/home/manhthe/project/data/original/%s/%s" % (fol,filename)) as img:
			print("Resizing %s....\n" % filename)
			w = img.width
			h = img.height
			#grayscale
			
			
			#resize
			r = float(128.0 / w)
			#dim = (128, int( float(r) * h ))
			img.resize(128, int(round( float(r) * h ,0)))
			#img.equalizeHist() #do sang tieu chuan
			img.type = "grayscale"
			img.save(filename="/home/manhthe/project/dulieu/%s/%d" % (fol,i) + ".jpg")
			i = i+1
video.release()
cv2.destroyAllWindows()
