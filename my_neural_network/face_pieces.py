import dlib
import cv2
import numpy
from imutils import face_utils
import math
from copy import copy

class Face:
    def __init__(self,image, detectorAndPredictor=None):   #assume one face image which detected via program
        if detectorAndPredictor is None:
            p = "shape_predictor_68_face_landmarks.dat"
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(p)
        else:
            detector, predictor = detectorAndPredictor
        self.image = cv2.resize(image,(280,320))
        self.shape = None
        self.face_part_lengths = None
        self.face_part_distances = None
        self.doted_face = None
        self.__setAttributes(detector,predictor)

    def __setAttributes(self,detector,predictor):
        #to understand what attributes have withinside check face_encoding.jpeg and view dot number
        #Note: since arrays starts with zero, subtract 1 from dot number to match its index with shapelist array
        self.__setShapesList(detector,predictor)
        self.__setDotedFace()
        self.__setFacePartLengths()
        self.__setFacePartDistances()


    def __setShapesList(self,detector,predictor):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        rect = detector(gray, 0)[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        self.shape = shape

    def __setDotedFace(self):
        self.doted_face = copy(self.image)
        for (x, y) in self.shape:
            cv2.circle(self.doted_face, (x, y), 2, (0, 255, 0), -1)

    def __setFacePartLengths(self):
        cheek = self.__getCheekLenght()
        left_brow = self.__getLeftBrorwLength()
        right_brow = self.__getRightBrorwLength()
        left_eye = self.__getLeftEyeLength()
        right_eye = self.__getRightEyeLength()
        nasal_bone = self.__getNasalBoneLength()
        nasal_tip = self.__getNasalTipLength()
        lip_outer = self.__getLipOuterLength()
        lip_inner = self.__getLipInnerLength()

        face_part_lengths = []
        face_part_lengths.append((cheek, "cheek"))
        face_part_lengths.append((left_brow, "left_brow"))
        face_part_lengths.append((right_brow, "right_brow"))
        face_part_lengths.append((left_eye, "left_eye"))
        face_part_lengths.append((right_eye, "right_eye"))
        face_part_lengths.append((nasal_bone, "nasal_bone"))
        face_part_lengths.append((nasal_tip, "nasal_tip"))
        face_part_lengths.append((lip_outer, "lip_outer"))
        face_part_lengths.append((lip_inner, "lip_inner"))

        self.face_part_lengths = face_part_lengths

    def __setFacePartDistances(self):
        left_brow_right_brow = self.__getLeftBrowRightBrowDistance()         #nearest points
        left_eye_left_brow = self.__getLeftEyeLeftBrowDistance()             #top
        right_eye_right_brow = self.__getRightEyeRightBrowDistance()         #top
        left_eye_right_eye = self.__getLeftEyeRightEyeDistance()             #nearest points
        left_eye_nasal_tip = self.__getLeftEyeNasalTipDistance()             #bottom
        right_eye_nasal_tip = self.__getRightRyeNasalTipDistance()           #bottom
        nasal_tip_lip_outer = self.__getNasalTipLipOuterDistance()           #top
        chin_lip_outer = self.__getChinLipOuterDistance()                    #bottom
        left_eye_left_whisker = self.__getLeftEyeLeftWhiskerDistance()       #nearest points
        right_eye_right_whisker = self.__getRightEyeRightWhiskerDistance()   #nearest points

        face_part_distances = []
        face_part_distances.append((left_brow_right_brow, "left_brow_right_brow"))
        face_part_distances.append((left_eye_left_brow, "left_eye_left_brow"))
        face_part_distances.append((right_eye_right_brow, "right_eye_right_brow"))
        face_part_distances.append((left_eye_right_eye, "left_eye_right_eye"))
        face_part_distances.append((left_eye_nasal_tip, "left_eye_nasal_tip"))
        face_part_distances.append((right_eye_nasal_tip, "right_eye_nasal_tip"))
        face_part_distances.append((nasal_tip_lip_outer, "nasal_tip_lip_outer"))
        face_part_distances.append((chin_lip_outer, "chin_lip_outer"))
        face_part_distances.append((left_eye_left_whisker, "left_eye_left_whisker"))
        face_part_distances.append((right_eye_right_whisker, "right_eye_right_whisker"))

        self.face_part_distances = face_part_distances

    def findDistance(self,point1,point2):
        x1, y1 = point1
        x2, y2 = point2
        x_diff = x1-x2
        y_diff = y1-y2
        return math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))

    def __getCheekLenght(self):
        sum = 0
        for i in range(1,17):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[16]
        point2 = self.shape[0]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum
    def __getLeftBrorwLength(self):
        sum = 0
        for i in range(18,22):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[21]
        point2 = self.shape[17]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum
    def __getRightBrorwLength(self):
        sum = 0
        for i in range(23,27):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[26]
        point2 = self.shape[22]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum
    def __getNasalBoneLength(self):
        sum = 0
        for i in range(28,31):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[30]
        point2 = self.shape[27]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum
    def __getNasalTipLength(self):
        sum = 0
        for i in range(32,36):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[35]
        point2 = self.shape[31]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum
    def __getLeftEyeLength(self):
        sum = 0
        for i in range(37,42):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[41]
        point2 = self.shape[36]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum
    def __getRightEyeLength(self):
        sum = 0
        for i in range(43,48):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[47]
        point2 = self.shape[42]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum
    def __getLipOuterLength(self):
        sum = 0
        for i in range(49,60):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[59]
        point2 = self.shape[48]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum
    def __getLipInnerLength(self):
        sum = 0
        for i in range(61,68):
            point1 = self.shape[i-1]
            point2 = self.shape[i]
            distance = self.findDistance(point1,point2)
            sum += distance
        point1 = self.shape[67]
        point2 = self.shape[60]
        distance = self.findDistance(point1, point2)
        sum += distance
        return sum

    def __getLeftBrowRightBrowDistance(self):
        point1 = self.shape[22]
        point2 = self.shape[23]
        return self.findDistance(point1, point2)
    def __getLeftEyeLeftBrowDistance(self):
        point1 = self.shape[20]
        point2 = self.shape[38]
        return self.findDistance(point1, point2)
    def __getRightEyeRightBrowDistance(self):
        point1 = self.shape[25]
        point2 = self.shape[45]
        return self.findDistance(point1, point2)
    def __getLeftEyeRightEyeDistance(self):
        point1 = self.shape[40]
        point2 = self.shape[43]
        return self.findDistance(point1, point2)
    def __getLeftEyeNasalTipDistance(self):
        point1 = self.shape[42]
        point2 = self.shape[34]
        return self.findDistance(point1, point2)
    def __getRightRyeNasalTipDistance(self):
        point1 = self.shape[47]
        point2 = self.shape[34]
        return self.findDistance(point1, point2)
    def __getNasalTipLipOuterDistance(self):
        point1 = self.shape[34]
        point2 = self.shape[52]
        return self.findDistance(point1, point2)
    def __getChinLipOuterDistance(self):
        point1 = self.shape[58]
        point2 = self.shape[9]
        return self.findDistance(point1, point2)
    def __getLeftEyeLeftWhiskerDistance(self):
        point1 = self.shape[18]
        point2 = self.shape[1]
        return self.findDistance(point1, point2)
    def __getRightEyeRightWhiskerDistance(self):
        point1 = self.shape[27]
        point2 = self.shape[17]
        return self.findDistance(point1, point2)