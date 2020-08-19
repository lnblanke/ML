# Find words in a picture
# @Time: 8/12/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Initialization.py

import cv2
import numpy as np
import os

# NMS 方法（Non Maximum Suppression，非极大值抑制）
def nms ( boxes , overlapThresh ) :
    if len ( boxes ) == 0 :
        return [ ]

    pick = [ ]

    # 取四个坐标数组
    x1 = boxes [ : , 0 ]
    y1 = boxes [ : , 1 ]
    x2 = boxes [ : , 2 ]
    y2 = boxes [ : , 3 ]

    # 计算面积数组
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按得分排序（如没有置信度得分，可按坐标从小到大排序，如右下角坐标）
    idxs = np.argsort ( y2 )

    # 开始遍历，并删除重复的框
    while len ( idxs ) > 0 :
        # 将最右下方的框放入pick数组
        last = len ( idxs ) - 1
        i = idxs [ last ]
        pick.append ( i )

        # 找剩下的其余框中最大坐标和最小坐标
        xx1 = np.maximum ( x1 [ i ] , x1 [ idxs [ :last ] ] )
        yy1 = np.maximum ( y1 [ i ] , y1 [ idxs [ :last ] ] )
        xx2 = np.minimum ( x2 [ i ] , x2 [ idxs [ :last ] ] )
        yy2 = np.minimum ( y2 [ i ] , y2 [ idxs [ :last ] ] )

        # 计算重叠面积占对应框的比例，即 IoU
        w = np.maximum ( 0 , xx2 - xx1 + 1 )
        h = np.maximum ( 0 , yy2 - yy1 + 1 )
        overlap = (w * h) / area [ idxs [ :last ] ]

        # 如果 IoU 大于指定阈值，则删除
        idxs = np.delete ( idxs , np.concatenate ( ([ last ] , np.where ( overlap > overlapThresh ) [ 0 ]) ) )

    return boxes [ pick ].astype ( "int" )

img = cv2.imread ( "test.jpg" )

hei , wei = img.shape [ 0 : 2 ]

img = cv2.resize ( img , ( 1000 , int ( hei / wei * 1000 ) ) )

copy = img

copy = cv2.cvtColor ( copy , cv2.COLOR_BGR2GRAY )

mser = cv2.MSER_create ()
region , _ = mser.detectRegions ( copy )
hulls = [ cv2.convexHull ( p.reshape ( -1 , 1 , 2 ) ) for p in region ]

keep = []

for c in hulls :
    x , y , w , h = cv2.boundingRect ( c )

    keep.append ( [ x , y , x + w , y + h ] )

box = nms ( np.array ( keep ) , 0.5 )

i = 0

for x1 , y1 , x2 , y2 in box :
    crop = img [ y1 : y2 , x1 : x2 ]

    crop = cv2.cvtColor ( crop , cv2.COLOR_BGR2GRAY )

    cv2.imwrite ( os.path.join ( "Testset" , str ( i ) + ".jpg") , crop )

    i += 1