from skimage import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import os.path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier



def createTrainData():
    mnist = fetch_mldata('MNIST original')
    if (os.path.exists(os.path.join(DIR, 'mnistPrepared') + '.npy')):
        train = np.load(os.path.join(DIR, 'mnistPrepared') + '.npy')
    else:
        train = mnist.data
        prepareTrainData(train)
        np.save(os.path.join(DIR, 'mnistPrepared'), train)
    train_labels = mnist.target
    return KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(train, train_labels)

def prepareTrainData(data):
    for i in range(0, len(data)):
        img = np.zeros((28, 28))
        number = data[i].reshape(28, 28)
        th = cv2.inRange(number, 140, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        labeled = label(closing)
        regions = regionprops(labeled)
        if(len(regions) == 1):
            bbox = regions[0].bbox
        else:
            max_width = 0
            max_height = 0
            for region in regions:
                t_bbox = region.bbox

                if(max_width < t_bbox[3] - t_bbox[1] and max_height < t_bbox[2] - t_bbox[0]):
                    max_height = t_bbox[2] - t_bbox[0]
                    max_width = t_bbox[3] - t_bbox[1]
                    bbox = t_bbox


        for x,row in enumerate(range(bbox[0], bbox[2])):
            for y,col in enumerate(range(bbox[1], bbox[3])):
                img[x, y] = number[row, col]

        data[i] = img.reshape(1, 784)

def lineEquation(x, y):
    return x*(y2-y1) + y*(x1-x2) + (x2*y1-x1*y2)

def getNumberImage(bbox):
    for x in range(0, len(img_number)):
        for y in range(0, len(img_number)):
            img_number[x, y] = gray[bbox[0]+x-1, bbox[1]+y-1]

def intersection(bbox):
    if(bbox[2]+4 < y2 or bbox[3]+4 < x1 or bbox[1] > x2):
        return False

    corners=[lineEquation(bbox[1], bbox[0]), lineEquation(bbox[3]+4, bbox[0]), lineEquation(bbox[1], bbox[2]+4), lineEquation(bbox[3]+4, bbox[2]+4)]
    recEnd = lineEquation(bbox[3], bbox[2])

    if all(cor>0 for cor in corners):
        return False
    elif all(cor<0 for cor in corners):
        return False
    elif(recEnd >=0 and frameNum >=1188):
        return False
    elif(recEnd<=0):
        return  False
    else:
        return True

def addNumber(broj, bbox, frameNum):
    x = bbox[1]
    y = bbox[0]
    for tup in reversed(lista_brojeva):
        isti=tup[0] == broj and tup[1] <= x+5 and tup[2] <= y+5 and tup[3] + 2 <= frameNum
        if isti:
            if frameNum >= tup[3]+250:
                break
            lista_brojeva.remove(tup)
            lista_brojeva.append((broj, x, y, frameNum))
            return False

    lista_brojeva.append((broj, x, y,frameNum))

DIR = 'C:\\Users\sale1\Desktop\Soft'
file = open(DIR+'\\out.txt','w')
file.write('RA 39/2014 Aleksandar Arambasic\nfile	sum\n')
knn=createTrainData()
img_number=np.zeros((28,28))

lista_brojeva=[]
DIR = DIR+'\\Videos'
video_names=[]
for name in os.listdir(DIR):
    if os.path.isfile(os.path.join(DIR, name)):
        video_names.append(os.path.join(DIR, name))
#video_names = [os.path.join(DIR, 'video-8.avi')]
for vid_num in range(0, len(video_names)):
    cap = cv2.VideoCapture(video_names[vid_num])
    frameNum = 0
    lista_brojeva.clear()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if(frameNum%2 != 0):
            frameNum += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        th = cv2.inRange(gray, 120, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        gray_labeled = label(closing)
        regions = regionprops(gray_labeled)

        if (frameNum == 0):

            line_th = cv2.inRange(gray, 10, 55)
            erosion = cv2.erode(line_th, np.ones((2, 2), np.uint8), iterations=1)
            skeleton = skeletonize(erosion / 255.0)
            cv_skeleton = img_as_ubyte(skeleton)
            lines = cv2.HoughLinesP(cv_skeleton, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            x1, y1, x2, y2 = lines[0][0]

        for region in regions:
            bbox = region.bbox
            if(bbox[2]-bbox[0] <= 10):
                continue
            if intersection(bbox) == False:
                continue

            getNumberImage(bbox)
            num = int(knn.predict(img_number.reshape(1, 784)))

            addNumber(num, bbox, frameNum)
        frameNum += 1
    suma=0

    for tup in lista_brojeva:
        suma += tup[0]
    print(video_names[vid_num])
    print('Suma: '+str(suma)+'\n')
    file.write('video-'+str(vid_num)+'.avi\t '+str(suma)+'\n')
cap.release()
cv2.destroyAllWindows()