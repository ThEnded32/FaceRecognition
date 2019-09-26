from PySide2.QtGui import QIcon
from PySide2 import QtWidgets, QtCore
from PySide2.QtWidgets import *
from functools import partial
import time
from library import *


class MainWindow():
    def initializeMainWindow(self):
        self.mainWindow =  QtWidgets.QWidget()
        self.mainWindow.setGeometry(50, 50, 300,300)
        selectImageToolButton = QtWidgets.QPushButton("Select Image",self.mainWindow)
        x=selectImageToolButton.geometry().width()
        y = selectImageToolButton.geometry().height()
        selectImageToolButton.move(150-(x//2),150-(y//2))
        selectImageToolButton.clicked.connect(self.imageSelect)


    def __init__(self):
        self.app = QApplication([])
        self.initializeMainWindow()
        self.initializeAttributes()


    def show(self):
        self.mainWindow.show()
        self.app.exec_()

    def initializeAttributes(self):
        self.neuralNetwork = NeuralNetwork()
        self.neuralNetwork.loadNeuralNetwork()
        self.createNewWindow()
        pass

    def imageSelect(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self.mainWindow, caption="Select Image File", dir = "",
                                                filter="All Files (*);;Image Files (*.jpg ; *.png ; *.jpeg )", options=options)
        if filepath and self.isImage(filepath):
            self.faceShow(filepath)
        else:
            QtWidgets.QMessageBox.about(self.mainWindow,"Not A Image File","Select Image File")


    def isImage(self, filepath):
        extensions = {".jpg", ".png", ".jpeg"}
        return any(filepath.endswith(ext) for ext in extensions)


    def createNewWindow(self,width=200,height=200):
        self.helperWindow = QtWidgets.QWidget()
        self.helperWindow.setGeometry(50, 50, width, height)


    def addButtonToShowFaceWindow(self,width,height,filepath):
        selectButton = QPushButton("Select Image",self.helperWindow)
        selectButton.resize(width,25)
        selectButton.move(0,height)
        selectButton.clicked.connect(partial(self.drawRectangleToFaces,filepath))
        returnButton = QPushButton("Return Main Window",self.helperWindow)
        returnButton.resize(width,25)
        returnButton.move(0,height+25)
        returnButton.clicked.connect(self.returnMainPage)


    def addButtonToDrawRectangleToFacesFaceWindow(self,width,height):
        button = QPushButton("Done",self.helperWindow)
        button.resize(width,25)
        button.move(0,height)
        button.clicked.connect(self.checkFaces)

    def checkFaces(self):

        self.setFaces()
        self.helperWindow.close()
        if(len(self.faces)>0):
            self.addNamesToFaces()
            self.showFacesByName()
            self.helperWindow.show()
        else:
            self.mainWindow.show()

    def faceShow(self,filepath):
        self.mainWindow.close()
        pixMap = QtGui.QPixmap(filepath)
        height = pixMap.size().height()
        width = pixMap.size().width()
        self.createNewWindow(width,height+50)
        self.mainWindow.close()
        self.addButtonToShowFaceWindow(width,height,filepath)
        label = QtWidgets.QLabel(self.helperWindow)
        label.setPixmap(pixMap)
        self.helperWindow.show()

    def drawRectangleToFaces(self,filepath):
        self.faces = []
        self.img = cv2.imread(filepath)
        self.imgRect,self.faceCoordinates = drawRectangleToFaces(self.img)
        pixMap = cv2ToPixmap(self.imgRect)
        height = pixMap.size().height()
        width = pixMap.size().width()
        self.createNewWindow(width,height+25)
        label = QtWidgets.QLabel(self.helperWindow)
        label.setPixmap(pixMap)
        self.addButtonToDrawRectangleToFacesFaceWindow(width,height)
        self.addButtonToFaces()
        self.helperWindow.show()
        self.showMessage("Face Selection","To Select A Face, Press Its Rectangle Area")



    def showMessage(self,title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        msg.buttonClicked.connect(msg.close)


    def addButtonToFaces(self):
        self.tempFaceCoordinates = []
        for coordinate in self.faceCoordinates:
            ((x1, y1), (x2, y2)) = coordinate
            cv2img = self.imgRect[y1:y2, x1:x2]
            pixMap = cv2ToPixmap(cv2img)
            height = pixMap.size().height()
            width = pixMap.size().width()
            icon = QIcon(pixMap)
            facebtn = QPushButton("", self.helperWindow)
            facebtn.resize(width, height)
            facebtn.clicked.connect(partial(self.addCoordinate,coordinate,facebtn))
            facebtn.setIcon(icon)
            facebtn.setIconSize(QtCore.QSize(width, height))
            facebtn.move(x1,y1)


    def addCoordinate(self,coordinate,facebtn):
        self.tempFaceCoordinates.append(coordinate)
        facebtn.hide()

    def setFaces(self):
        for coordinate in self.tempFaceCoordinates:
            ((x1, y1), (x2, y2)) = coordinate
            cv2img = self.img[y1:y2, x1:x2]
            self.faces.append(cv2.resize(cv2img,(80,100)))



    def addNamesToFaces(self):
        predictions = self.neuralNetwork.getPredictions(self.faces)
        namesAndRatios = self.neuralNetwork.getPredictionNamesAndRatios(predictions)
        for i,face in enumerate(self.faces):
            if namesAndRatios[i][1]<0.8:name = "Unknown"
            else: name = namesAndRatios[i][0]
            self.faces[i] = (face,name)
    """

    def chanceFacesFormat(self):
        for i,(face,name) in enumerate(self.faces):
            self.faces[i]=(cv2ToPixmap(face),name)
    """

    def showFacesByName(self):
        boxsize =(400,100)
        width = boxsize[0]
        height = boxsize[1]
        (rowNumber,columnNumber) = self.getRowAndColumnNumber()
        self.helperWindow.close()
        self.createNewWindow(width*columnNumber,height*rowNumber+25)
        self.helperWindow.setGeometry(50,50,width*columnNumber,height*rowNumber+25)
        self.showFaces(rowNumber,columnNumber,boxsize)
        returnbtn = QPushButton("Return Main Window",self.helperWindow)
        returnbtn.clicked.connect(self.returnMainPage)
        returnbtn.resize(width*columnNumber,25)
        returnbtn.move(0,height*rowNumber)

    def returnMainPage(self):
        self.helperWindow.close()
        self.mainWindow.show()

    def getRowAndColumnNumber(self):
        number = len(self.faces)
        if number>8:
            return (8,(number//8)+1)
        else:
            return (number,1)

    def showFaces(self, rowNumber, columnNumber, boxsize):
        currentRow = 0
        currentColumn = 0
        for i,(face,name) in enumerate(self.faces):
            label = QLabel(self.helperWindow)
            label.setPixmap(cv2ToPixmap(face))
            coordinate=(boxsize[0]*currentColumn,boxsize[1]*currentRow)
            label.move(coordinate[0],coordinate[1])
            label = QLabel(self.helperWindow)
            label.setText(name)
            label.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Black))
            label.move(coordinate[0]+80,coordinate[1]+40)
            """            
            if(name == "Unknown"):
                self.addButtonToUnknown(i,coordinate)
            """
            if(i+1)%8 == 0:
                currentColumn+=1
                currentRow=0
            else:
                currentRow+=1
    """
    def addButtonToUnknown(self, i, coordinate):
        pixMap = cv2ToPixmap(self.faces[i][0])
        height = pixMap.size().height()
        width = pixMap.size().width()
        icon = QIcon(pixMap)
        facebtn = QPushButton("", self.helperWindow)
        facebtn.resize(width, height)
        facebtn.clicked.connect(lambda : print("printed"))
        facebtn.setIcon(icon)
        facebtn.setIconSize(QtCore.QSize(width, height))
        facebtn.move(coordinate[0],coordinate[1])
    """





