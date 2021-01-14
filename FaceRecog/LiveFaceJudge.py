# Judge whether the face in the camera is the person in the dataset
# @Time: 8/17/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: LiveFaceJudge.py

import cv2 , dlib , numpy , time

detector = dlib.get_frontal_face_detector ()
predictor = dlib.shape_predictor ( "shape_predictor_68_face_landmarks.dat" )
model = dlib.face_recognition_model_v1 ( "dlib_face_recognition_resnet_model_v1.dat" )

global desp
pic = cv2.imread ( "Dataset/obama.jpg" )

faces = detector ( pic , 1 )

for i , face in enumerate ( faces ) :
    shape = predictor ( pic , face )

    descriptor = model.compute_face_descriptor ( pic , shape )

    vec = numpy.array ( descriptor )

    desp = vec

cap = cv2.VideoCapture ( 0 )
_ , img = cap.read ()

faces = detector ( img , 1 )

for i , face in enumerate ( faces ) :
    shape = predictor ( img , face )

    descriptor = model.compute_face_descriptor ( img , shape )

    vect = numpy.array ( descriptor )

    d = numpy.linalg.norm ( desp - vect )

    if d < 0.7 :
        print ( "Correct!" )
    else :
        print ( "Incorrect!" )
