import cv2
import numpy as np
import os
import dlib
from face_pieces import Face

def getFacePoints(image):
    path = "xmlfiles/haarcascade_frontalface_default.xml"
    face_cas = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return face_cas.detectMultiScale(gray, 1.1, 4)

def getAdvancedFacePoints(image):
    advanced_points = []
    for (x,y,w,h) in getFacePoints(image):
        x2 = x+w
        y2 = int(y+ h*1.2)
        max = image.shape[1]-1
        if(y2>max):
            y2=max
        advanced_points.append(((y,x),(y2,x2)))
    return advanced_points

def getFaces(image):
    faces = []
    for point1,point2 in getAdvancedFacePoints(image):
        x1,y1 = point1
        x2,y2 = point2
        face = image[x1:x2,y1:y2]
        faces.append(face)
    return faces

def getFace(image):
    return getFaces(image)[0]

def getImageFilePaths(path):
    imagePaths = []
    extensions = [".jpg", ".png",".jpeg"]
    for cwd, subdir, filenames in os.walk(path):
        for filename in filenames:
            extension = os.path.splitext(filename)[1]
            if extension in extensions:
                imagePath = os.path.join(cwd, filename)
                imagePaths.append(imagePath)
    return  imagePaths

def getDataSet(train):   #hardcoded:special func for my dataset
    #use only first face
    sub_path = "test"
    if train:
        sub_path = "train"

    path = "face_dataset/"+sub_path+"/"
    dataArray = []
    for i in range(1,20):
        personno = "person{}".format(i)
        person = (i,getImageFilePaths(path+personno))
        dataArray.append(person)
    return dataArray

def saveFaceToTxt(train):
    sub_path = "testtxt/"
    if train:
        sub_path = "traintxt/"

    path = "/home/thended/PycharmProjects/opencv-projects/xmlfiles/haarcascade_frontalface_default.xml"
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    persons = getDataSet(train)

    for person in persons:
        paths = person[1]
        txtpath = "face_dataset/"+sub_path+str(person[0])+".txt"
        file = open(txtpath, "w")
        for path in paths:
            image = cv2.imread(path)
            face = getFace(image)
            face = Face(face,(detector,predictor))
            face_array =face.face_part_lengths+face.face_part_distances
            file.write(",".join(str(x[0]) for x in face_array)+"\n")
        file.close()

def getFaceDatas(train):
    sub_path = "face_dataset/testtxt/person{}.txt"
    if train:
        sub_path = "face_dataset/traintxt/person{}.txt"

    all_data = []

    for i in range(1,20):
        path = sub_path.format(i)
        file = open(path,"r")
        face_datas = []
        for line in file.readlines():
            line = line[0:-2]
            face_datas.append([float(x) for x in line.split(",")] )
        all_data.append((i,face_datas))
    return all_data

def saveFaces(train):
    sub_path = "face_dataset/testfaces/person{}/"
    if train:
        sub_path = "face_dataset/trainfaces/person{}/"

    data = getDataSet(train)
    for personno,paths in data:
        new_path = sub_path.format(personno)
        counter = 1
        for path in paths:
            img = cv2.imread(path)
            face = getFace(img)
            face = cv2.resize(face, (140, 160))
            cv2.imwrite(new_path+"face{}.jpg".format(counter),face)
            counter += 1

def getGrayFacesForTF(train):
    sub_path = "face_dataset/testfaces/person{}/"
    if train:
        sub_path = "face_dataset/trainfaces/person{}/"
    data = []
    for personno in range(1,20):
        file_path = sub_path.format(personno)
        paths = getImageFilePaths(file_path)
        faces = []
        for path in paths:
            img = cv2.imread(path,0)
            img = np.array(img,dtype=np.float64)
            faces.append(img)
        data.append((personno,faces))
    return data