# Create verification code with random 4 digits
# @Time: 8/19/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: createCode.py

from captcha.image import ImageCaptcha
import numpy as np
import os

set = [ '0' , '1' , '2' , '3' , '4' , '5' , '6' , '7' , '8' , '9' ]

for i in range ( 1000 ) :
    rd = np.random.rand ( 4 ) * 10

    text = ""

    for i in rd :
        text += set [ int ( i ) ]

    img = ImageCaptcha ()
    img.write ( text , os.path.join ( "Code Training" , text + ".jpg" ) )