# Find features of the face in the camera
# @Time: 8/17/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: LiveRecog.py

import cv2 , dlib

detector = dlib.get_frontal_face_detector ()
predictor = dlib.shape_predictor ( "shape_predictor_68_face_landmarks.dat" )

cap = cv2.VideoCapture ( 0 )

while 1 :
    _ , frame = cap.read ()

    grey = cv2.cvtColor ( frame , cv2.COLOR_BGR2GRAY )

    faces = detector ( grey )

    for face in faces :
        landmarks = predictor ( grey , face )

        for i in range ( 0 , 68 ) :
            x = landmarks.part ( i ).x
            y = landmarks.part ( i ).y

            cv2.circle ( frame , ( x , y ) , 3 , ( 0 , 255 , 0 ) , -1 )

        cv2.imshow ( "Face" , frame )

    if cv2.waitKey ( 1 ) == 27 :
        break
    else :
        continue

cap.release ()
cv2.destroyAllWindows ()
