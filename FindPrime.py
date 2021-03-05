# @Time: 2/5/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: FindPrime.py

import time
import matplotlib.pyplot as plot

initTime = time.time ()

dict = {}

n = 2

while n < 1e5 :
    flag = 0

    for num in dict :
        if dict [ num ] == n :
            flag = 1
            dict [ num ] += num

    if not flag :
        dict [ n ] = 2 * n

        plot.scatter ( n , time.time() - initTime , s = 2 , edgecolors = "b" )

    n += 1

plot.show ()