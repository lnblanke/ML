# Training with dataset and recognize the face in the live camera
# @Time: 8/17/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: LiveFace.py

import cv2 , dlib , os , numpy

detector = dlib.get_frontal_face_detector ()
predictor = dlib.shape_predictor ( "shape_predictor_68_face_landmarks.dat" )
model = dlib.face_recognition_model_v1 ( "dlib_face_recognition_resnet_model_v1.dat" )

desp = [ ]

for _ , _ , files in os.walk ( "Dataset" ) :
    for file in files :
        pic = cv2.imread ( os.path.join ( "Dataset" , file ) )

        faces = detector ( pic , 1 )

        for i , face in enumerate ( faces ) :
            shape = predictor ( pic , face )

            descriptor = model.compute_face_descriptor ( pic , shape )

            vec = numpy.array ( descriptor )

            desp.append ( vec )

cap = cv2.VideoCapture ( 0 )
_ , img = cap.read ()

faces = detector ( img , 1 )

dist = [ ]
for i , face in enumerate ( faces ) :
    shape = predictor ( img , face )

    descriptor = model.compute_face_descriptor ( img , shape )

    vect = numpy.array ( descriptor )

    for iter in desp :
        d = numpy.linalg.norm ( iter - vect )

        dist.append ( d )

candidate = [ "Trump" , "Obama" , "Biden" , "Clinton" , "David" ]

diction = dict ( zip ( candidate , dist ) )

diction = sorted ( diction.items () , key = lambda d : d [ 1 ] )
print ( "This person is: " , diction [ 0 ] [ 0 ] )
