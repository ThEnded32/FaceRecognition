import os
import cv2
import numpy as np
from PySide2 import QtGui
from PySide2.QtGui import QImage
import tensorflow as tf

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


def getName(path):
    splited = path.split("/")
    return splited[-2]

def getFaceImages(paths):
    faceImages=[]
    for path in paths:
        img = FaceImage(path)
        faceImages.append(img)
    return faceImages

def getFaceCoordinates(image):
    path = "xmlfiles/haarcascade_frontalface_default.xml"
    face_cas = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cas.detectMultiScale(gray, 1.1, 4)

def getAdvancedFaceCoordinates(image):
    advanced_points = []
    for (x,y,w,h) in getFaceCoordinates(image):
        x2 = x+w
        y2 = int(y+ h*1.2)
        max = image.shape[1]-1
        if(y2>max):
            y2=max
        advanced_points.append(((x,y),(x2,y2)))
    return advanced_points

def getFaces(image):
    faces = []
    for ((x1,y1), (x2,y2)) in getAdvancedFaceCoordinates(image):
        face = image[y1:y2,x1:x2]
        faces.append(face)
    return faces

def drawRectangleToFaces(image):
    drawImage =cv2.copyMakeBorder(image,0,0,0,0,cv2.BORDER_REPLICATE)
    faceCoordinates = getAdvancedFaceCoordinates(drawImage)
    for (point1,point2) in faceCoordinates:
        cv2.rectangle(drawImage, point1, point2, (255,0,0), 2)
    return (drawImage,faceCoordinates)

def cv2ToPixmap(cv2img):
    img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qImg)
    return pixmap

def getGrayImage(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def splitArray(array,ratio=0.8,shuffle=True):   #to split datasets to train and test
    big = []
    small = []
    copyArray = array.copy()
    if shuffle: np.random.shuffle(copyArray)
    for x in copyArray:
        if np.random.random()<=ratio:
            big.append(x)
        else:
            small.append(x)
    return (big,small)


class FaceImage:        #container class for face attributes
    def __init__(self,path):
        self.path = path
        self.image =np.array( cv2.imread(self.path))

    def getName(self):
        return getName(self.path)
    def getPath(self):
        return self.path
    def getImage(self):
        return self.image
    def getGrayImage(self):
        return getGrayImage(self.image)
    def getFace(self):
        return getFaces(self.image)[0]
    def getGrayFace(self):
        return getGrayImage(self.getFace())
    def getFaceForNeuralNetwork(self):
        return cv2.resize(self.getGrayFace(),(70,80))

class NeuralNetwork:        #tensorflow used
    def __init__(self):
        self.faceImages=getFaceImages(getImageFilePaths("faceDataset"))
        self.modelName = "NeuralNetwork.model"
        self.setNameDictionary()

    def createNeuralNetwork(self):
        size = len(self.nameDictionary)
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(size*4, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(size*4, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(size, activation=tf.nn.softmax))
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def saveNeuralNetwork(self):
        self.model.save(self.modelName)

    def loadNeuralNetwork(self):
        self.model = tf.keras.models.load_model(self.modelName)

    #if FullTrain is true use all FaceImages to train otherwise use nearly 80%
    def trainNeuralNetwork(self,duplicate=False,epochs = 25):

        if duplicate:
            trainX,trainY = self.getImageByNameNumber(self.faceImages.copy())
            test = self.faceImages.copy()
            np.random.shuffle(test)
            testX,testY = self.getImageByNameNumber(test[:len(test)//5])
        else:
            train,test = splitArray(self.faceImages)
            trainX, trainY = self.getImageByNameNumber(train)
            testX, testY = self.getImageByNameNumber(test)

        trainX = np.array(trainX, dtype=np.float64)
        trainX = tf.keras.utils.normalize(trainX, axis=1)
        trainY = np.array(trainY, dtype=np.float64)
        testX = np.array(testX, dtype=np.float64)
        testX = tf.keras.utils.normalize(testX, axis=1)
        testY = np.array(testY, dtype=np.float64)
        self.model.fit(trainX, trainY, epochs=epochs)

        return self.model.evaluate(testX,testY)


    def setNameDictionary(self):
        nameArray = []
        for face in self.faceImages:
            name = face.getName()
            if name not in nameArray:
                nameArray.append(name)
        nameDictionary = {}
        for i,name in enumerate(nameArray):
            nameDictionary[name]=i
        self.nameDictionary = nameDictionary

    def getImageByNameNumber(self,faceImages):
        image = []
        name = []

        for face in faceImages:
            image.append(face.getFaceForNeuralNetwork())
            name.append(self.nameDictionary[face.getName()])
        return image,name

    def getPredictions(self,faceList):
        faces = []
        for face in faceList:
            img = getGrayImage(face)
            img = cv2.resize(img,(70,80))
            faces.append(np.array(img))

        faces = np.array(faces, dtype=np.float64)
        faces = tf.keras.utils.normalize(faces, axis=1)
        return self.model.predict(faces)

    def getPredictionNamesAndRatios(self,predictions):
        mydict = self.nameDictionary
        namesAndRatios = []
        for prediction in predictions:
            index = np.argmax(prediction)
            name = list(mydict.keys())[list(mydict.values()).index(index)]
            ratio = prediction[index]
            namesAndRatios.append((name,ratio))
        return namesAndRatios