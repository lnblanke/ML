# Cut videos into series of pictures
# @Time: 3/27/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Video Cutting.py

import cv2

def getVideoCut ( source ) :
    video = cv2.VideoCapture ( source )
    rval , frame = video.read ()

    images = []

    time = 0

    while rval :
        rval , frame = video.read ()

        if time % 30000 == 0 :
            images.append ( frame )

        time += 1000

    return images